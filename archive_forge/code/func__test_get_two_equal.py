import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def _test_get_two_equal(self, pyfunc):
    """
        Test with two arrays of the same type
        """
    rec = numpy_support.from_dtype(recordtype)
    cfunc = self.get_cfunc(pyfunc, (rec[:], rec[:], types.intp))
    for i in range(self.refsample1d.size):
        self.assertEqual(pyfunc(self.refsample1d, self.refsample1d3, i), cfunc(self.nbsample1d, self.nbsample1d3, i))