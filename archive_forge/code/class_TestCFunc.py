import ctypes
import os
import subprocess
import sys
from collections import namedtuple
import numpy as np
from numba import cfunc, carray, farray, njit
from numba.core import types, typing, utils
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import (TestCase, skip_unless_cffi, tag,
import unittest
from numba.np import numpy_support
class TestCFunc(TestCase):

    def test_basic(self):
        """
        Basic usage and properties of a cfunc.
        """
        f = cfunc(add_sig)(add_usecase)
        self.assertEqual(f.__name__, 'add_usecase')
        self.assertEqual(f.__qualname__, 'add_usecase')
        self.assertIs(f.__wrapped__, add_usecase)
        symbol = f.native_name
        self.assertIsInstance(symbol, str)
        self.assertIn('add_usecase', symbol)
        addr = f.address
        self.assertIsInstance(addr, int)
        ct = f.ctypes
        self.assertEqual(ctypes.cast(ct, ctypes.c_void_p).value, addr)
        self.assertPreciseEqual(ct(2.0, 3.5), 5.5)

    @skip_unless_cffi
    def test_cffi(self):
        from numba.tests import cffi_usecases
        ffi, lib = cffi_usecases.load_inline_module()
        f = cfunc(square_sig)(square_usecase)
        res = lib._numba_test_funcptr(f.cffi)
        self.assertPreciseEqual(res, 2.25)

    def test_locals(self):
        f = cfunc(div_sig, locals={'c': types.int64})(div_usecase)
        self.assertPreciseEqual(f.ctypes(8, 3), 2.0)

    def test_errors(self):
        f = cfunc(div_sig)(div_usecase)
        with captured_stderr() as err:
            self.assertPreciseEqual(f.ctypes(5, 2), 2.5)
        self.assertEqual(err.getvalue(), '')
        with captured_stderr() as err:
            res = f.ctypes(5, 0)
            self.assertPreciseEqual(res, 0.0)
        err = err.getvalue()
        self.assertIn('ZeroDivisionError:', err)
        self.assertIn('Exception ignored', err)

    def test_llvm_ir(self):
        f = cfunc(add_sig)(add_usecase)
        ir = f.inspect_llvm()
        self.assertIn(f.native_name, ir)
        self.assertIn('fadd double', ir)

    def test_object_mode(self):
        """
        Object mode is currently unsupported.
        """
        with self.assertRaises(NotImplementedError):
            cfunc(add_sig, forceobj=True)(add_usecase)
        with self.assertTypingError() as raises:
            cfunc(add_sig)(objmode_usecase)
        self.assertIn("Untyped global name 'object'", str(raises.exception))