import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
class TestNdZeros(ConstructorBaseTest, TestCase):

    def setUp(self):
        super(TestNdZeros, self).setUp()
        self.pyfunc = np.zeros

    def check_result_value(self, ret, expected):
        np.testing.assert_equal(ret, expected)

    def test_0d(self):
        pyfunc = self.pyfunc

        def func():
            return pyfunc(())
        self.check_0d(func)

    def test_1d(self):
        pyfunc = self.pyfunc

        def func(n):
            return pyfunc(n)
        self.check_1d(func)

    def test_1d_dtype(self):
        pyfunc = self.pyfunc

        def func(n):
            return pyfunc(n, np.int32)
        self.check_1d(func)

    def test_1d_dtype_instance(self):
        pyfunc = self.pyfunc
        _dtype = np.dtype('int32')

        def func(n):
            return pyfunc(n, _dtype)
        self.check_1d(func)

    def test_1d_dtype_str(self):
        pyfunc = self.pyfunc
        _dtype = 'int32'

        def func(n):
            return pyfunc(n, _dtype)
        self.check_1d(func)

        def func(n):
            return pyfunc(n, 'complex128')
        self.check_1d(func)

    def test_1d_dtype_str_alternative_spelling(self):
        pyfunc = self.pyfunc
        _dtype = 'i4'

        def func(n):
            return pyfunc(n, _dtype)
        self.check_1d(func)

        def func(n):
            return pyfunc(n, 'c8')
        self.check_1d(func)

    def test_1d_dtype_str_structured_dtype(self):
        pyfunc = self.pyfunc
        _dtype = 'i4, (2,3)f8'

        def func(n):
            return pyfunc(n, _dtype)
        self.check_1d(func)

    def test_1d_dtype_non_const_str(self):
        pyfunc = self.pyfunc

        @njit
        def func(n, dt):
            return pyfunc(n, dt)
        with self.assertRaises(TypingError) as raises:
            func(5, 'int32')
        excstr = str(raises.exception)
        msg = f'If np.{self.pyfunc.__name__} dtype is a string it must be a string constant.'
        self.assertIn(msg, excstr)

    def test_1d_dtype_invalid_str(self):
        pyfunc = self.pyfunc

        @njit
        def func(n):
            return pyfunc(n, 'ABCDEF')
        with self.assertRaises(TypingError) as raises:
            func(5)
        excstr = str(raises.exception)
        self.assertIn("Invalid NumPy dtype specified: 'ABCDEF'", excstr)

    def test_2d(self):
        pyfunc = self.pyfunc

        def func(m, n):
            return pyfunc((m, n))
        self.check_2d(func)

    def test_2d_shape_dtypes(self):
        pyfunc = self.pyfunc

        def func1(m, n):
            return pyfunc((np.int16(m), np.int32(n)))
        self.check_2d(func1)

        def func2(m, n):
            return pyfunc((np.int64(m), np.int8(n)))
        self.check_2d(func2)
        if config.IS_32BITS:
            cfunc = nrtjit(lambda m, n: pyfunc((m, n)))
            with self.assertRaises(ValueError):
                cfunc(np.int64(1 << 32 - 1), 1)

    def test_2d_dtype_kwarg(self):
        pyfunc = self.pyfunc

        def func(m, n):
            return pyfunc((m, n), dtype=np.complex64)
        self.check_2d(func)

    def test_2d_dtype_str_kwarg(self):
        pyfunc = self.pyfunc

        def func(m, n):
            return pyfunc((m, n), dtype='complex64')
        self.check_2d(func)

    def test_2d_dtype_str_kwarg_alternative_spelling(self):
        pyfunc = self.pyfunc

        def func(m, n):
            return pyfunc((m, n), dtype='c8')
        self.check_2d(func)

    def test_alloc_size(self):
        pyfunc = self.pyfunc
        width = types.intp.bitwidth

        def gen_func(shape, dtype):
            return lambda: pyfunc(shape, dtype)
        self.check_alloc_size(gen_func(1 << width - 2, np.intp))
        self.check_alloc_size(gen_func((1 << width - 8, 64), np.intp))