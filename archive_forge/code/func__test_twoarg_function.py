from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
import numpy as np
def _test_twoarg_function(self, f):
    x = np.asarray((10, 9, 8, 7, 6))
    y = np.asarray((1, 2, 3, 4, 5))
    error = np.zeros(1, dtype=np.int32)
    f[1, 1](x, y, error)
    self.assertEqual(error[0], 0)