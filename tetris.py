# Authors: yuanming-hu and k-ye

import taichi as ti
import numpy as np
import random
import os

write_to_disk = True
ti.init(arch=ti.gpu)  # Try to run on GPU

quality = 1  # Use a larger value for higher-res simulations
n_grid = (40 * quality, 80 * quality)
dx, inv_dx = 1 / n_grid[1], n_grid[1]
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 0.15e4, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters

max_num_particles = 1024 * 16
dim = 2
x = ti.Vector.field(dim, dtype=float)  # position
v = ti.Vector.field(dim, dtype=float)  # velocity
C = ti.Matrix.field(dim, dim, dtype=float)  # affine velocity field
F = ti.Matrix.field(dim, dim, dtype=float)  # deformation gradient
material = ti.field(dtype=int)  # material id
Jp = ti.field(dtype=float)  # plastic deformation

ti.root.dynamic(ti.i, max_num_particles).place(x, v, C, F, material, Jp)
cur_num_particles = ti.field(ti.i32, shape=())

grid_v = ti.Vector.field(dim, dtype=float,
                         shape=n_grid)  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=n_grid)  # grid node mass


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

    # p2g
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F[p] = (ti.Matrix.identity(float, 2) +
                dt * C[p]) @ F[p]  # deformation gradient update
        h = ti.exp(
            10 *
            (1.0 -
             Jp[p]))  # Hardening coefficient: snow gets harder when compressed
        if material[p] >= 2:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 1:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                              1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[
                p] == 0:  # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[p] == 1:
            F[p] = U @ sig @ V.transpose(
            )  # Reconstruct elastic deformation gradient after plasticity
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose(
        ) + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(
                3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i,
                   j] = (1 / grid_m[i, j]) * grid_v[i,
                                                    j]  # Momentum to velocity
            grid_v[i, j][1] -= dt * 50  # gravity
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid[0] - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j] = ti.Vector([0, 0])
            if j > n_grid[1] - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0

    # g2p
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(
                3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection


num_per_tetromino_square = 128
num_per_tetromino = num_per_tetromino_square * 4
staging_tetromino_x = ti.Vector.field(dim,
                                      dtype=float,
                                      shape=num_per_tetromino)


class StagingTetromino(object):
    def __init__(self, x):
        """
        Stores the info of the staging tetromino.

        Args
          x: a taichi field. Note this class just keeps |x| as a reference, and
             is not a taichi data_oriented class.
        """
        self._x_field = x
        self.material_idx = 0

        self._offsets = np.array([
            [[0, -1], [1, 0], [0, -2]],
            [[1, 1], [-1, 0], [1, 0]],
            [[0, -1], [-1, 0], [0, -2]],
            [[0, 1], [1, 0], [1, -1]],
            [[1, 0], [2, 0], [-1, 0]],
            [[0, 1], [1, 1], [1, 0]],
            [[-1, 0], [1, 0], [0, 1]],
        ])
        self._x_np_canonical = None
        self.left_width = 0
        self.right_width = 0
        self.lower_height = 0
        self.upper_height = 0

    def regenerate(self, mat, kind):
        self.material_idx = mat
        shape = (num_per_tetromino, dim)
        x = np.zeros(shape=shape, dtype=np.float32)
        for i in range(1, 4):
            # is there a more np-idiomatic way?
            begin = i * num_per_tetromino_square
            x[begin:(begin +
                     num_per_tetromino_square)] = self._offsets[kind, (i - 1)]
        x += np.random.rand(*shape)
        scaling = 0.05
        x *= scaling
        self._x_np_canonical = x

        self.left_width= scaling * abs(min(self._offsets[kind, :][:, 0]))
        self.right_width= scaling * abs(max(self._offsets[kind, :][:, 0]) + 1)
        self.lower_height = scaling * abs(min(self._offsets[kind, :][:, 1]))
        self.upper_height = scaling * abs(max(self._offsets[kind, :][:, 1]) + 1)
        

    def update_center(self, center):
        self._x_field.from_numpy(np.clip(self._x_np_canonical + center, 0, 1))

    def rotate(self):
        theta = np.radians(90)
        c, s = np.cos(theta), np.sin(theta)
        m = np.array([[c, -s], [s, c]], dtype=np.float32)
        x = m @ self._x_np_canonical.T
        self._x_np_canonical = x.T

        self.right_width, self.lower_height, self.left_width, self.upper_height = \
            self.lower_height, self.left_width, self.upper_height, self.right_width

    def compute_center(self, mouse, l_bound, r_bound):
        r = staging_tetromino.right_width
        l = staging_tetromino.left_width

        if mouse[0] + r > r_bound:
            x = r_bound - r
        elif mouse[0] - l < l_bound:
            x = l_bound + l
        else:
            x = mouse[0]

        return np.array([x, 0.8], dtype=np.float32)

staging_tetromino = StagingTetromino(staging_tetromino_x)


@ti.kernel
def drop_staging_tetromino(mat: int):
    base = cur_num_particles[None]
    for i in staging_tetromino_x:
        bi = base + i
        x[bi] = staging_tetromino_x[i]
        material[bi] = mat
        v[bi] = ti.Matrix([0, -2])
        F[bi] = ti.Matrix([[1, 0], [0, 1]])
        Jp[bi] = 1
    cur_num_particles[None] += num_per_tetromino


def main():
    os.makedirs('frames', exist_ok=True)
    gui = ti.GUI("Taichi MLS-MPM-99",
                 res=(384, 768),
                 background_color=0x112F41)

    def gen_mat_and_kind():
        material_id = random.randint(0, 7)
        return material_id, random.randint(0, 6)

    staging_tetromino.regenerate(*gen_mat_and_kind())

    last_action_frame = -1e10
    for f in range(100000):
        padding = 0.025
        segments = 20
        step = (1 - padding * 4) / (segments - 0.5) / 2
        for i in range(segments):
            gui.line(begin=(padding * 2 + step * 2 * i, 0.8),
                     end=(padding * 2 + step * (2 * i + 1), 0.8),
                     radius=1.5,
                     color=0xFF8811)
        gui.line(begin=(padding * 2, padding),
                 end=(1 - padding * 2, padding),
                 radius=2)
        gui.line(begin=(padding * 2, 1 - padding),
                 end=(1 - padding * 2, 1 - padding),
                 radius=2)
        gui.line(begin=(padding * 2, padding),
                 end=(padding * 2, 1 - padding),
                 radius=2)
        gui.line(begin=(1 - padding * 2, padding),
                 end=(1 - padding * 2, 1 - padding),
                 radius=2)

        if gui.get_event(ti.GUI.PRESS):
            ev_key = gui.event.key
            if ev_key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: break
            elif ev_key == ti.GUI.SPACE:
                if cur_num_particles[
                        None] + num_per_tetromino < max_num_particles:
                    drop_staging_tetromino(staging_tetromino.material_idx)
                print('# particles =', cur_num_particles[None])
                staging_tetromino.regenerate(*gen_mat_and_kind())
                last_action_frame = f
            elif ev_key == 'r':
                staging_tetromino.rotate()
        mouse = gui.get_cursor_pos()
        mouse = (mouse[0] * 0.5, mouse[1])

        right_bound = 0.5 - padding
        left_bound = padding
        
        staging_tetromino.update_center(staging_tetromino.compute_center(mouse, left_bound, right_bound))

        for s in range(int(2e-3 // dt)):
            substep()
        colors = np.array([
            0xA6B5F7, 0xEEEEF0, 0xED553B, 0x3255A7, 0x6D35CB, 0xFE2E44,
            0x26A5A7, 0xEDE53B
        ],
                          dtype=np.uint32)
        particle_radius = 2.3
        gui.circles(x.to_numpy() * [[2, 1]],
                    radius=particle_radius,
                    color=colors[material.to_numpy()])

        if last_action_frame + 40 < f:
            gui.circles(staging_tetromino_x.to_numpy() * [[2, 1]],
                        radius=particle_radius,
                        color=int(colors[staging_tetromino.material_idx]))

            if staging_tetromino.material_idx == 0:
                mat_text = 'Liquid'
            elif staging_tetromino.material_idx == 1:
                mat_text = 'Snow'
            else:
                mat_text = 'Jelly'
            gui.text(mat_text, (0.42, 0.97),
                     font_size=30,
                     color=colors[staging_tetromino.material_idx])
        gui.text('Taichi Tetris', (0.07, 0.97), font_size=20)

        
        if write_to_disk:
            gui.show(f'frames/{f:05d}.png')
        else:
            gui.show()


if __name__ == '__main__':
    main()
