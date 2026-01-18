import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
def _test_polyarithm_basic(self, pyfunc, ignore_sign_on_zero=False):
    cfunc = njit(pyfunc)

    def inputs():
        for i in range(5):
            for j in range(5):
                p1 = np.array([0] * i + [1])
                p2 = np.array([0] * j + [1])
                yield (p1, p2)
        yield ([1, 2, 3], [1, 2, 3])
        yield ([1, 2, 3], (1, 2, 3))
        yield ((1, 2, 3), [1, 2, 3])
        yield ([1, 2, 3], 3)
        yield (3, (1, 2, 3))
        yield (np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]))
        yield (np.array([1j, 2j, 3j]), np.array([1.0, 2.0, 3.0]))
        yield (np.array([1, 2, 3]), np.array([1j, 2j, 3j]))
        yield ((1, 2, 3), 3.0)
        yield ((1, 2, 3), 3j)
        yield ((1, 0.001, 3), (1, 2, 3))
    for p1, p2 in inputs():
        self.assertPreciseEqual(pyfunc(p1, p2), cfunc(p1, p2), ignore_sign_on_zero=ignore_sign_on_zero)