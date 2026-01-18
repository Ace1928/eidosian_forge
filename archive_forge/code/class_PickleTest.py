import unittest
import pickle
from ctypes import *
import _ctypes_test
class PickleTest:

    def dumps(self, item):
        return pickle.dumps(item, self.proto)

    def loads(self, item):
        return pickle.loads(item)

    def test_simple(self):
        for src in [c_int(42), c_double(3.14)]:
            dst = self.loads(self.dumps(src))
            self.assertEqual(src.__dict__, dst.__dict__)
            self.assertEqual(memoryview(src).tobytes(), memoryview(dst).tobytes())

    def test_struct(self):
        X.init_called = 0
        x = X()
        x.a = 42
        self.assertEqual(X.init_called, 1)
        y = self.loads(self.dumps(x))
        self.assertEqual(X.init_called, 1)
        self.assertEqual(y.__dict__, x.__dict__)
        self.assertEqual(memoryview(y).tobytes(), memoryview(x).tobytes())

    def test_unpickable(self):
        self.assertRaises(ValueError, lambda: self.dumps(Y()))
        prototype = CFUNCTYPE(c_int)
        for item in [c_char_p(), c_wchar_p(), c_void_p(), pointer(c_int(42)), dll._testfunc_p_p, prototype(lambda: 42)]:
            self.assertRaises(ValueError, lambda: self.dumps(item))

    def test_wchar(self):
        self.dumps(c_char(b'x'))
        self.dumps(c_wchar('x'))