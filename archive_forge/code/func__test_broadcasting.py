import numpy as np
from numba import float32, jit, njit
from numba.np.ufunc import Vectorize
from numba.core.errors import TypingError
from numba.tests.support import TestCase
import unittest
def _test_broadcasting(self, cls, a, b, c, d):
    """Test multiple args"""
    vectorizer = cls(add_multiple_args)
    vectorizer.add(float32(float32, float32, float32, float32))
    ufunc = vectorizer.build_ufunc()
    info = (cls, a.shape)
    self.assertPreciseEqual(ufunc(a, b, c, d), a + b + c + d, msg=info)