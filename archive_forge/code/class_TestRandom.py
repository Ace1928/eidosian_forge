import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestRandom(CUDATestCase):

    def test_ex_3d_grid(self):
        from numba import cuda
        from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
        import numpy as np

        @cuda.jit
        def random_3d(arr, rng_states):
            startx, starty, startz = cuda.grid(3)
            stridex, stridey, stridez = cuda.gridsize(3)
            tid = startz * stridey * stridex + starty * stridex + startx
            for i in range(startz, arr.shape[0], stridez):
                for j in range(starty, arr.shape[1], stridey):
                    for k in range(startx, arr.shape[2], stridex):
                        arr[i, j, k] = xoroshiro128p_uniform_float32(rng_states, tid)
        X, Y, Z = (701, 900, 719)
        bx, by, bz = (8, 8, 8)
        gx, gy, gz = (16, 16, 16)
        nthreads = bx * by * bz * gx * gy * gz
        rng_states = create_xoroshiro128p_states(nthreads, seed=1)
        arr = cuda.device_array((X, Y, Z), dtype=np.float32)
        random_3d[(gx, gy, gz), (bx, by, bz)](arr, rng_states)
        host_arr = arr.copy_to_host()
        self.assertGreater(np.mean(host_arr), 0.49)
        self.assertLess(np.mean(host_arr), 0.51)
        self.assertTrue(np.all(host_arr <= 1.0))
        self.assertTrue(np.all(host_arr >= 0.0))