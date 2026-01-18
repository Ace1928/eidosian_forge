from ctypes import *
import sys
import threading
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.core.typing import ctypes_utils
from numba.tests.support import MemoryLeakMixin, tag, TestCase
from numba.tests.ctypes_usecases import *
import unittest
def _conversion_tests(self, check):
    check(c_double, types.float64)
    check(c_int, types.intc)
    check(c_uint16, types.uint16)
    check(c_size_t, types.size_t)
    check(c_ssize_t, types.ssize_t)
    check(c_void_p, types.voidptr)
    check(POINTER(c_float), types.CPointer(types.float32))
    check(POINTER(POINTER(c_float)), types.CPointer(types.CPointer(types.float32)))
    check(None, types.void)