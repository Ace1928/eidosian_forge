import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
class TestRecarray(CUDATestCase):

    def test_recarray(self):
        a = np.recarray((16,), dtype=[('value1', np.int64), ('value2', np.float64)])
        a.value1 = np.arange(a.size, dtype=np.int64)
        a.value2 = np.arange(a.size, dtype=np.float64) / 100
        expect1 = a.value1
        expect2 = a.value2

        def test(x, out1, out2):
            i = cuda.grid(1)
            if i < x.size:
                out1[i] = x.value1[i]
                out2[i] = x.value2[i]
        got1 = np.zeros_like(expect1)
        got2 = np.zeros_like(expect2)
        cuda.jit(test)[1, a.size](a, got1, got2)
        np.testing.assert_array_equal(expect1, got1)
        np.testing.assert_array_equal(expect2, got2)