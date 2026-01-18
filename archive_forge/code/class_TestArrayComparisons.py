from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
class TestArrayComparisons(TestCase):

    def test_identity(self):

        def check(a, b, expected):
            cfunc = njit((typeof(a), typeof(b)))(pyfunc)
            self.assertPreciseEqual(cfunc(a, b), (expected, not expected))
        pyfunc = identity_usecase
        arr = np.zeros(10, dtype=np.int32).reshape((2, 5))
        check(arr, arr, True)
        check(arr, arr[:], True)
        check(arr, arr.copy(), False)
        check(arr, arr.view('uint32'), False)
        check(arr, arr.T, False)
        check(arr, arr[:-1], False)