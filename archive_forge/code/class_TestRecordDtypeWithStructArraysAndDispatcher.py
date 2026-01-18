import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
class TestRecordDtypeWithStructArraysAndDispatcher(TestRecordDtypeWithStructArrays):
    """
    Same as TestRecordDtypeWithStructArrays, stressing the Dispatcher's type
    dispatch mechanism (issue #384) and caching of ndarray typecodes for void
    types (which occur in structured arrays).
    """

    def get_cfunc(self, pyfunc, argspec):
        return _get_cfunc_nopython(pyfunc, argspec)