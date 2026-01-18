import itertools
import pickle
import textwrap
import numpy as np
from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc
def _check_reduce(self, ufunc, dtype=None, initial=None):

    @njit
    def foo(a, axis, dtype, initial):
        return ufunc.reduce(a, axis=axis, dtype=dtype, initial=initial)
    inputs = [np.arange(5), np.arange(4).reshape(2, 2), np.arange(40).reshape(5, 4, 2)]
    for array in inputs:
        for axis in range(array.ndim):
            expected = foo.py_func(array, axis, dtype, initial)
            got = foo(array, axis, dtype, initial)
            self.assertPreciseEqual(expected, got)