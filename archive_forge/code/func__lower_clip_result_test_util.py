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
def _lower_clip_result_test_util(self, func, a, a_min, a_max):

    def lower_clip_result(a):
        return np.expm1(func(a, a_min, a_max))
    np.testing.assert_almost_equal(lower_clip_result(a), jit(nopython=True)(lower_clip_result)(a))