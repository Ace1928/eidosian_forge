import ctypes
import unittest
from numba.core import types
from numba.core.extending import intrinsic
from numba import jit, njit
from numba.tests.support import captured_stdout
def PyBytes_AsStringAndSize(uni):
    return _pyapi_bytes_as_string_and_size(uni._data, uni._length)