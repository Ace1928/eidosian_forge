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
class TestNdFull(ConstructorBaseTest, TestCase):

    def check_result_value(self, ret, expected):
        np.testing.assert_equal(ret, expected)

    def test_0d(self):

        def func():
            return np.full((), 4.5)
        self.check_0d(func)

    def test_1d(self):

        def func(n):
            return np.full(n, 4.5)
        self.check_1d(func)

    def test_1d_dtype(self):

        def func(n):
            return np.full(n, 4.5, np.bool_)
        self.check_1d(func)

    def test_1d_dtype_instance(self):
        dtype = np.dtype('bool')

        def func(n):
            return np.full(n, 4.5, dtype)
        self.check_1d(func)

    def test_1d_dtype_str(self):

        def func(n):
            return np.full(n, 4.5, 'bool_')
        self.check_1d(func)

    def test_1d_dtype_str_alternative_spelling(self):

        def func(n):
            return np.full(n, 4.5, '?')
        self.check_1d(func)

    def test_1d_dtype_non_const_str(self):

        @njit
        def func(n, fv, dt):
            return np.full(n, fv, dt)
        with self.assertRaises(TypingError) as raises:
            func((5,), 4.5, 'int32')
        excstr = str(raises.exception)
        msg = 'If np.full dtype is a string it must be a string constant.'
        self.assertIn(msg, excstr)

    def test_1d_dtype_invalid_str(self):

        @njit
        def func(n, fv):
            return np.full(n, fv, 'ABCDEF')
        with self.assertRaises(TypingError) as raises:
            func((5,), 4.5)
        excstr = str(raises.exception)
        self.assertIn("Invalid NumPy dtype specified: 'ABCDEF'", excstr)

    def test_2d(self):

        def func(m, n):
            return np.full((m, n), 4.5)
        self.check_2d(func)

    def test_2d_dtype_kwarg(self):

        def func(m, n):
            return np.full((m, n), 1 + 4.5j, dtype=np.complex64)
        self.check_2d(func)

    def test_2d_dtype_from_type(self):

        def func(m, n):
            return np.full((m, n), np.int32(1))
        self.check_2d(func)

        def func(m, n):
            return np.full((m, n), np.complex128(1))
        self.check_2d(func)

        def func(m, n):
            return np.full((m, n), 1, dtype=np.int8)
        self.check_2d(func)

    def test_2d_shape_dtypes(self):

        def func1(m, n):
            return np.full((np.int16(m), np.int32(n)), 4.5)
        self.check_2d(func1)

        def func2(m, n):
            return np.full((np.int64(m), np.int8(n)), 4.5)
        self.check_2d(func2)
        if config.IS_32BITS:
            cfunc = nrtjit(lambda m, n: np.full((m, n), 4.5))
            with self.assertRaises(ValueError):
                cfunc(np.int64(1 << 32 - 1), 1)

    def test_alloc_size(self):
        width = types.intp.bitwidth

        def gen_func(shape, value):
            return lambda: np.full(shape, value)
        self.check_alloc_size(gen_func(1 << width - 2, 1))
        self.check_alloc_size(gen_func((1 << width - 8, 64), 1))