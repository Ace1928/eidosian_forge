import numpy as np
from numba import float32, float64, int32, uint32
from numba.np.ufunc import Vectorize
import unittest
class TestParallelLowWorkCount(unittest.TestCase):
    _numba_parallel_test_ = False

    def test_low_workcount(self):
        pv = Vectorize(vector_add, target='parallel')
        for ty in (int32, uint32, float32, float64):
            pv.add(ty(ty, ty))
        para_ufunc = pv.build_ufunc()
        np_ufunc = np.vectorize(vector_add)

        def test(ty):
            data = np.arange(1).astype(ty)
            result = para_ufunc(data, data)
            gold = np_ufunc(data, data)
            np.testing.assert_allclose(gold, result)
        test(np.double)
        test(np.float32)
        test(np.int32)
        test(np.uint32)