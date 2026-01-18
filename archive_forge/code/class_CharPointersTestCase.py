from ctypes import *
from ctypes.test import need_symbol
import unittest
import _ctypes_test
class CharPointersTestCase(unittest.TestCase):

    def setUp(self):
        func = testdll._testfunc_p_p
        func.restype = c_long
        func.argtypes = None

    def test_paramflags(self):
        prototype = CFUNCTYPE(c_void_p, c_void_p)
        func = prototype(('_testfunc_p_p', testdll), ((1, 'input'),))
        try:
            func()
        except TypeError as details:
            self.assertEqual(str(details), "required argument 'input' missing")
        else:
            self.fail('TypeError not raised')
        self.assertEqual(func(None), None)
        self.assertEqual(func(input=None), None)

    def test_int_pointer_arg(self):
        func = testdll._testfunc_p_p
        if sizeof(c_longlong) == sizeof(c_void_p):
            func.restype = c_longlong
        else:
            func.restype = c_long
        self.assertEqual(0, func(0))
        ci = c_int(0)
        func.argtypes = (POINTER(c_int),)
        self.assertEqual(positive_address(addressof(ci)), positive_address(func(byref(ci))))
        func.argtypes = (c_char_p,)
        self.assertRaises(ArgumentError, func, byref(ci))
        func.argtypes = (POINTER(c_short),)
        self.assertRaises(ArgumentError, func, byref(ci))
        func.argtypes = (POINTER(c_double),)
        self.assertRaises(ArgumentError, func, byref(ci))

    def test_POINTER_c_char_arg(self):
        func = testdll._testfunc_p_p
        func.restype = c_char_p
        func.argtypes = (POINTER(c_char),)
        self.assertEqual(None, func(None))
        self.assertEqual(b'123', func(b'123'))
        self.assertEqual(None, func(c_char_p(None)))
        self.assertEqual(b'123', func(c_char_p(b'123')))
        self.assertEqual(b'123', func(c_buffer(b'123')))
        ca = c_char(b'a')
        self.assertEqual(ord(b'a'), func(pointer(ca))[0])
        self.assertEqual(ord(b'a'), func(byref(ca))[0])

    def test_c_char_p_arg(self):
        func = testdll._testfunc_p_p
        func.restype = c_char_p
        func.argtypes = (c_char_p,)
        self.assertEqual(None, func(None))
        self.assertEqual(b'123', func(b'123'))
        self.assertEqual(None, func(c_char_p(None)))
        self.assertEqual(b'123', func(c_char_p(b'123')))
        self.assertEqual(b'123', func(c_buffer(b'123')))
        ca = c_char(b'a')
        self.assertEqual(ord(b'a'), func(pointer(ca))[0])
        self.assertEqual(ord(b'a'), func(byref(ca))[0])

    def test_c_void_p_arg(self):
        func = testdll._testfunc_p_p
        func.restype = c_char_p
        func.argtypes = (c_void_p,)
        self.assertEqual(None, func(None))
        self.assertEqual(b'123', func(b'123'))
        self.assertEqual(b'123', func(c_char_p(b'123')))
        self.assertEqual(None, func(c_char_p(None)))
        self.assertEqual(b'123', func(c_buffer(b'123')))
        ca = c_char(b'a')
        self.assertEqual(ord(b'a'), func(pointer(ca))[0])
        self.assertEqual(ord(b'a'), func(byref(ca))[0])
        func(byref(c_int()))
        func(pointer(c_int()))
        func((c_int * 3)())

    @need_symbol('c_wchar_p')
    def test_c_void_p_arg_with_c_wchar_p(self):
        func = testdll._testfunc_p_p
        func.restype = c_wchar_p
        func.argtypes = (c_void_p,)
        self.assertEqual(None, func(c_wchar_p(None)))
        self.assertEqual('123', func(c_wchar_p('123')))

    def test_instance(self):
        func = testdll._testfunc_p_p
        func.restype = c_void_p

        class X:
            _as_parameter_ = None
        func.argtypes = (c_void_p,)
        self.assertEqual(None, func(X()))
        func.argtypes = None
        self.assertEqual(None, func(X()))