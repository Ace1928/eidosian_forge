import numpy as np
from numba import cuda, complex64, int32, float64
from numba.cuda.testing import unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
class TestCudaConstantMemory(CUDATestCase):

    def test_const_array(self):
        sig = (float64[:],)
        jcuconst = cuda.jit(sig)(cuconst)
        A = np.zeros_like(CONST1D)
        jcuconst[2, 5](A)
        self.assertTrue(np.all(A == CONST1D + 1))
        if not ENABLE_CUDASIM:
            self.assertIn('ld.const.f64', jcuconst.inspect_asm(sig), "as we're adding to it, load as a double")

    def test_const_empty(self):
        jcuconstEmpty = cuda.jit('void(int64[:])')(cuconstEmpty)
        A = np.full(1, fill_value=-1, dtype=np.int64)
        jcuconstEmpty[1, 1](A)
        self.assertTrue(np.all(A == 0))

    def test_const_align(self):
        jcuconstAlign = cuda.jit('void(float64[:])')(cuconstAlign)
        A = np.full(3, fill_value=np.nan, dtype=float)
        jcuconstAlign[1, 3](A)
        self.assertTrue(np.all(A == CONST3BYTES + CONST1D[:3]))

    def test_const_array_2d(self):
        sig = (int32[:, :],)
        jcuconst2d = cuda.jit(sig)(cuconst2d)
        A = np.zeros_like(CONST2D, order='C')
        jcuconst2d[(2, 2), (5, 5)](A)
        self.assertTrue(np.all(A == CONST2D))
        if not ENABLE_CUDASIM:
            self.assertIn('ld.const.u32', jcuconst2d.inspect_asm(sig), 'load the ints as ints')

    def test_const_array_3d(self):
        sig = (complex64[:, :, :],)
        jcuconst3d = cuda.jit(sig)(cuconst3d)
        A = np.zeros_like(CONST3D, order='F')
        jcuconst3d[1, (5, 5, 5)](A)
        self.assertTrue(np.all(A == CONST3D))
        if not ENABLE_CUDASIM:
            asm = jcuconst3d.inspect_asm(sig)
            complex_load = 'ld.const.v2.f32'
            description = 'Load the complex as a vector of 2x f32'
            self.assertIn(complex_load, asm, description)

    def test_const_record_empty(self):
        jcuconstRecEmpty = cuda.jit('void(int64[:])')(cuconstRecEmpty)
        A = np.full(1, fill_value=-1, dtype=np.int64)
        jcuconstRecEmpty[1, 1](A)
        self.assertTrue(np.all(A == 0))

    def test_const_record(self):
        A = np.zeros(2, dtype=float)
        B = np.zeros(2, dtype=int)
        jcuconst = cuda.jit(cuconstRec).specialize(A, B)
        jcuconst[2, 1](A, B)
        np.testing.assert_allclose(A, CONST_RECORD['x'])
        np.testing.assert_allclose(B, CONST_RECORD['y'])

    def test_const_record_align(self):
        A = np.zeros(2, dtype=np.float64)
        B = np.zeros(2, dtype=np.float64)
        C = np.zeros(2, dtype=np.float64)
        D = np.zeros(2, dtype=np.float64)
        E = np.zeros(2, dtype=np.float64)
        jcuconst = cuda.jit(cuconstRecAlign).specialize(A, B, C, D, E)
        jcuconst[2, 1](A, B, C, D, E)
        np.testing.assert_allclose(A, CONST_RECORD_ALIGN['a'])
        np.testing.assert_allclose(B, CONST_RECORD_ALIGN['b'])
        np.testing.assert_allclose(C, CONST_RECORD_ALIGN['x'])
        np.testing.assert_allclose(D, CONST_RECORD_ALIGN['y'])
        np.testing.assert_allclose(E, CONST_RECORD_ALIGN['z'])