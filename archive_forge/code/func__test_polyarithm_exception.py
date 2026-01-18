import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
def _test_polyarithm_exception(self, pyfunc):
    cfunc = njit(pyfunc)
    self.disable_leak_check()
    with self.assertRaises(TypingError) as raises:
        cfunc('abc', np.array([1, 2, 3]))
    self.assertIn('The argument "c1" must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(np.array([1, 2, 3]), 'abc')
    self.assertIn('The argument "c2" must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as e:
        cfunc(np.arange(10).reshape(5, 2), np.array([1, 2, 3]))
    self.assertIn('Coefficient array is not 1-d', str(e.exception))
    with self.assertRaises(TypingError) as e:
        cfunc(np.array([1, 2, 3]), np.arange(10).reshape(5, 2))
    self.assertIn('Coefficient array is not 1-d', str(e.exception))