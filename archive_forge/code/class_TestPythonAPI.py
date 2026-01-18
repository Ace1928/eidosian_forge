import ctypes
import unittest
from numba.core import types
from numba.core.extending import intrinsic
from numba import jit, njit
from numba.tests.support import captured_stdout
class TestPythonAPI(unittest.TestCase):

    def test_PyBytes_AsString(self):
        cfunc = jit(nopython=True)(PyBytes_AsString)
        cstr = cfunc('hello')
        fn = ctypes.pythonapi.PyBytes_FromString
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.py_object
        obj = fn(cstr)
        self.assertEqual(obj, b'hello')

    def test_PyBytes_AsStringAndSize(self):
        cfunc = jit(nopython=True)(PyBytes_AsStringAndSize)
        tup = cfunc('hello\x00world')
        fn = ctypes.pythonapi.PyBytes_FromStringAndSize
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        fn.restype = ctypes.py_object
        obj = fn(tup[0], tup[1])
        self.assertEqual(obj, b'hello\x00world')