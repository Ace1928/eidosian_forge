import unittest
from test import support
import ctypes
import gc
import _ctypes_test
class AnotherLeak(unittest.TestCase):

    def test_callback(self):
        import sys
        proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)

        def func(a, b):
            return a * b * 2
        f = proto(func)
        a = sys.getrefcount(ctypes.c_int)
        f(1, 2)
        self.assertEqual(sys.getrefcount(ctypes.c_int), a)

    @support.refcount_test
    def test_callback_py_object_none_return(self):
        for FUNCTYPE in (ctypes.CFUNCTYPE, ctypes.PYFUNCTYPE):
            with self.subTest(FUNCTYPE=FUNCTYPE):

                @FUNCTYPE(ctypes.py_object)
                def func():
                    return None
                for _ in range(10000):
                    func()