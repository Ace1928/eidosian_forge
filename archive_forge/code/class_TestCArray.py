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
class TestCArray(TestCase):
    """
    Tests for carray() and farray().
    """

    def run_carray_usecase(self, pointer_factory, func):
        a = np.arange(10, 16).reshape((2, 3)).astype(np.float32)
        out = np.empty(CARRAY_USECASE_OUT_LEN, dtype=np.float32)
        func(pointer_factory(a), pointer_factory(out), *a.shape)
        return out

    def check_carray_usecase(self, pointer_factory, pyfunc, cfunc):
        expected = self.run_carray_usecase(pointer_factory, pyfunc)
        got = self.run_carray_usecase(pointer_factory, cfunc)
        self.assertPreciseEqual(expected, got)

    def make_voidptr(self, arr):
        return arr.ctypes.data_as(ctypes.c_void_p)

    def make_float32_pointer(self, arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def make_float64_pointer(self, arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def check_carray_farray(self, func, order):

        def eq(got, expected):
            self.assertPreciseEqual(got, expected)
            self.assertEqual(got.ctypes.data, expected.ctypes.data)
        base = np.arange(6).reshape((2, 3)).astype(np.float32).copy(order=order)
        a = func(self.make_float32_pointer(base), base.shape)
        eq(a, base)
        a = func(self.make_float32_pointer(base), base.size)
        eq(a, base.ravel('K'))
        a = func(self.make_float32_pointer(base), base.shape, base.dtype)
        eq(a, base)
        a = func(self.make_float32_pointer(base), base.shape, np.float32)
        eq(a, base)
        a = func(self.make_voidptr(base), base.shape, base.dtype)
        eq(a, base)
        a = func(self.make_voidptr(base), base.shape, np.int32)
        eq(a, base.view(np.int32))
        with self.assertRaises(TypeError):
            func(self.make_voidptr(base), base.shape)
        with self.assertRaises(TypeError):
            func(base.ctypes.data, base.shape)
        with self.assertRaises(TypeError) as raises:
            func(self.make_float32_pointer(base), base.shape, np.int32)
        self.assertIn("mismatching dtype 'int32' for pointer", str(raises.exception))

    def test_carray(self):
        """
        Test pure Python carray().
        """
        self.check_carray_farray(carray, 'C')

    def test_farray(self):
        """
        Test pure Python farray().
        """
        self.check_carray_farray(farray, 'F')

    def make_carray_sigs(self, formal_sig):
        """
        Generate a bunch of concrete signatures by varying the width
        and signedness of size arguments (see issue #1923).
        """
        for actual_size in (types.intp, types.int32, types.intc, types.uintp, types.uint32, types.uintc):
            args = tuple((actual_size if a == types.intp else a for a in formal_sig.args))
            yield formal_sig.return_type(*args)

    def check_numba_carray_farray(self, usecase, dtype_usecase):
        pyfunc = usecase
        for sig in self.make_carray_sigs(carray_float32_usecase_sig):
            f = cfunc(sig)(pyfunc)
            self.check_carray_usecase(self.make_float32_pointer, pyfunc, f.ctypes)
        pyfunc = dtype_usecase
        for sig in self.make_carray_sigs(carray_float32_usecase_sig):
            f = cfunc(sig)(pyfunc)
            self.check_carray_usecase(self.make_float32_pointer, pyfunc, f.ctypes)
        with self.assertTypingError() as raises:
            f = cfunc(carray_float64_usecase_sig)(pyfunc)
        self.assertIn("mismatching dtype 'float32' for pointer type 'float64*'", str(raises.exception))
        pyfunc = dtype_usecase
        for sig in self.make_carray_sigs(carray_voidptr_usecase_sig):
            f = cfunc(sig)(pyfunc)
            self.check_carray_usecase(self.make_float32_pointer, pyfunc, f.ctypes)

    def test_numba_carray(self):
        """
        Test Numba-compiled carray() against pure Python carray()
        """
        self.check_numba_carray_farray(carray_usecase, carray_dtype_usecase)

    def test_numba_farray(self):
        """
        Test Numba-compiled farray() against pure Python farray()
        """
        self.check_numba_carray_farray(farray_usecase, farray_dtype_usecase)