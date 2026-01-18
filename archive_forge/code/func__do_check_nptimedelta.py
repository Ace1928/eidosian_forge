from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def _do_check_nptimedelta(self, pyfunc, arr):
    arrty = typeof(arr)
    cfunc = jit(nopython=True)(pyfunc)
    self.assertPreciseEqual(cfunc(arr), pyfunc(arr))
    self.assertPreciseEqual(cfunc(arr[:-1]), pyfunc(arr[:-1]))
    arr = arr[::-1].copy()
    self.assertPreciseEqual(cfunc(arr), pyfunc(arr))
    np.random.shuffle(arr)
    self.assertPreciseEqual(cfunc(arr), pyfunc(arr))
    if 'median' not in pyfunc.__name__:
        for x in range(1, len(arr), 2):
            arr[x] = 'NaT'
        self.assertPreciseEqual(cfunc(arr), pyfunc(arr))
    arr.fill(arrty.dtype('NaT'))
    self.assertPreciseEqual(cfunc(arr), pyfunc(arr))