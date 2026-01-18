import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
class WAP(types.WrapperAddressProtocol):
    """An example implementation of wrapper address protocol.

    """

    def __init__(self, func, sig):
        self.pyfunc = func
        self.cfunc = cfunc(sig)(func)
        self.sig = sig

    def __wrapper_address__(self):
        return self.cfunc._wrapper_address

    def signature(self):
        return self.sig

    def __call__(self, *args, **kwargs):
        return self.pyfunc(*args, **kwargs)