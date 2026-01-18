import numpy as np
from numba import cuda
from numba.core.config import ENABLE_CUDASIM
from numba.cuda.testing import CUDATestCase
import unittest
def _sum_reduce(self, n):
    A = np.arange(n, dtype=np.float64) + 1
    expect = A.sum()
    got = sum_reduce(A)
    self.assertEqual(expect, got)