import sys
import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.tests.support import captured_stdout, TestCase
from numba.np import numpy_support
def _setup_usecase2to5(self, dtype):
    N = 5
    a = np.recarray(N, dtype=dtype)
    a.f1 = np.arange(N)
    a.f2 = np.arange(2, N + 2)
    a.s1 = np.array(['abc'] * a.shape[0], dtype='|S3')
    return a