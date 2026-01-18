import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
class Bad(types.WrapperAddressProtocol):
    """A first-class function type with invalid 0 address.
            """

    def __wrapper_address__(self):
        return 0

    def signature(self):
        return sig