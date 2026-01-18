import itertools
import pickle
import textwrap
import numpy as np
from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc
def _check_reduce_axis(self, ufunc, dtype, initial=None):

    @njit
    def foo(a, axis):
        return ufunc.reduce(a, axis=axis, initial=initial)

    def _check(*args):
        try:
            expected = foo.py_func(array, axis)
        except ValueError as e:
            self.assertEqual(e.args[0], exc_msg)
            with self.assertRaisesRegex(TypingError, exc_msg):
                got = foo(array, axis)
        else:
            got = foo(array, axis)
            self.assertPreciseEqual(expected, got)
    exc_msg = f"reduction operation '{ufunc.__name__}' is not reorderable, so at most one axis may be specified"
    inputs = [np.arange(40, dtype=dtype).reshape(5, 4, 2), np.arange(10, dtype=dtype)]
    for array in inputs:
        for i in range(1, array.ndim + 1):
            for axis in itertools.combinations(range(array.ndim), r=i):
                _check(array, axis)
        for axis in ((), None):
            _check(array, axis)