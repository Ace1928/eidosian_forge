import functools
import unittest
from test import support
from ctypes import *
from ctypes.test import need_symbol
from _ctypes import CTYPES_MAX_ARGCOUNT
import _ctypes_test
class SampleCallbacksTestCase(unittest.TestCase):

    def test_integrate(self):
        dll = CDLL(_ctypes_test.__file__)
        CALLBACK = CFUNCTYPE(c_double, c_double)
        integrate = dll.integrate
        integrate.argtypes = (c_double, c_double, CALLBACK, c_long)
        integrate.restype = c_double

        def func(x):
            return x ** 2
        result = integrate(0.0, 1.0, CALLBACK(func), 10)
        diff = abs(result - 1.0 / 3.0)
        self.assertLess(diff, 0.01, '%s not less than 0.01' % diff)

    def test_issue_8959_a(self):
        from ctypes.util import find_library
        libc_path = find_library('c')
        if not libc_path:
            self.skipTest('could not find libc')
        libc = CDLL(libc_path)

        @CFUNCTYPE(c_int, POINTER(c_int), POINTER(c_int))
        def cmp_func(a, b):
            return a[0] - b[0]
        array = (c_int * 5)(5, 1, 99, 7, 33)
        libc.qsort(array, len(array), sizeof(c_int), cmp_func)
        self.assertEqual(array[:], [1, 5, 7, 33, 99])

    @need_symbol('WINFUNCTYPE')
    def test_issue_8959_b(self):
        from ctypes.wintypes import BOOL, HWND, LPARAM
        global windowCount
        windowCount = 0

        @WINFUNCTYPE(BOOL, HWND, LPARAM)
        def EnumWindowsCallbackFunc(hwnd, lParam):
            global windowCount
            windowCount += 1
            return True
        windll.user32.EnumWindows(EnumWindowsCallbackFunc, 0)

    def test_callback_register_int(self):
        dll = CDLL(_ctypes_test.__file__)
        CALLBACK = CFUNCTYPE(c_int, c_int, c_int, c_int, c_int, c_int)
        func = dll._testfunc_cbk_reg_int
        func.argtypes = (c_int, c_int, c_int, c_int, c_int, CALLBACK)
        func.restype = c_int

        def callback(a, b, c, d, e):
            return a + b + c + d + e
        result = func(2, 3, 4, 5, 6, CALLBACK(callback))
        self.assertEqual(result, callback(2 * 2, 3 * 3, 4 * 4, 5 * 5, 6 * 6))

    def test_callback_register_double(self):
        dll = CDLL(_ctypes_test.__file__)
        CALLBACK = CFUNCTYPE(c_double, c_double, c_double, c_double, c_double, c_double)
        func = dll._testfunc_cbk_reg_double
        func.argtypes = (c_double, c_double, c_double, c_double, c_double, CALLBACK)
        func.restype = c_double

        def callback(a, b, c, d, e):
            return a + b + c + d + e
        result = func(1.1, 2.2, 3.3, 4.4, 5.5, CALLBACK(callback))
        self.assertEqual(result, callback(1.1 * 1.1, 2.2 * 2.2, 3.3 * 3.3, 4.4 * 4.4, 5.5 * 5.5))

    def test_callback_large_struct(self):

        class Check:
            pass

        class X(Structure):
            _fields_ = [('first', c_ulong), ('second', c_ulong), ('third', c_ulong)]

        def callback(check, s):
            check.first = s.first
            check.second = s.second
            check.third = s.third
            s.first = s.second = s.third = 195948557
        check = Check()
        s = X()
        s.first = 3735928559
        s.second = 3405691582
        s.third = 195894762
        CALLBACK = CFUNCTYPE(None, X)
        dll = CDLL(_ctypes_test.__file__)
        func = dll._testfunc_cbk_large_struct
        func.argtypes = (X, CALLBACK)
        func.restype = None
        func(s, CALLBACK(functools.partial(callback, check)))
        self.assertEqual(check.first, s.first)
        self.assertEqual(check.second, s.second)
        self.assertEqual(check.third, s.third)
        self.assertEqual(check.first, 3735928559)
        self.assertEqual(check.second, 3405691582)
        self.assertEqual(check.third, 195894762)
        self.assertEqual(s.first, check.first)
        self.assertEqual(s.second, check.second)
        self.assertEqual(s.third, check.third)

    def test_callback_too_many_args(self):

        def func(*args):
            return len(args)
        proto = CFUNCTYPE(c_int, *(c_int,) * CTYPES_MAX_ARGCOUNT)
        cb = proto(func)
        args1 = (1,) * CTYPES_MAX_ARGCOUNT
        self.assertEqual(cb(*args1), CTYPES_MAX_ARGCOUNT)
        args2 = (1,) * (CTYPES_MAX_ARGCOUNT + 1)
        with self.assertRaises(ArgumentError):
            cb(*args2)
        with self.assertRaises(ArgumentError):
            CFUNCTYPE(c_int, *(c_int,) * (CTYPES_MAX_ARGCOUNT + 1))

    def test_convert_result_error(self):

        def func():
            return ('tuple',)
        proto = CFUNCTYPE(c_int)
        ctypes_func = proto(func)
        with support.catch_unraisable_exception() as cm:
            result = ctypes_func()
            self.assertIsInstance(cm.unraisable.exc_value, TypeError)
            self.assertEqual(cm.unraisable.err_msg, 'Exception ignored on converting result of ctypes callback function')
            self.assertIs(cm.unraisable.object, func)