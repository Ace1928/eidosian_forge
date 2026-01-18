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
class TestNdEmptyLike(ConstructorLikeBaseTest, TestCase):

    def setUp(self):
        super(TestNdEmptyLike, self).setUp()
        self.pyfunc = np.empty_like

    def check_result_value(self, ret, expected):
        pass

    def test_like(self):
        pyfunc = self.pyfunc

        def func(arr):
            return pyfunc(arr)
        self.check_like(func, np.float64)

    def test_like_structured(self):
        dtype = np.dtype([('a', np.int16), ('b', np.float32)])
        pyfunc = self.pyfunc

        def func(arr):
            return pyfunc(arr)
        self.check_like(func, dtype)

    def test_like_dtype(self):
        pyfunc = self.pyfunc

        def func(arr):
            return pyfunc(arr, np.int32)
        self.check_like(func, np.float64)

    def test_like_dtype_instance(self):
        dtype = np.dtype('int32')
        pyfunc = self.pyfunc

        def func(arr):
            return pyfunc(arr, dtype)
        self.check_like(func, np.float64)

    def test_like_dtype_structured(self):
        dtype = np.dtype([('a', np.int16), ('b', np.float32)])
        pyfunc = self.pyfunc

        def func(arr):
            return pyfunc(arr, dtype)
        self.check_like(func, np.float64)

    def test_like_dtype_kwarg(self):
        pyfunc = self.pyfunc

        def func(arr):
            return pyfunc(arr, dtype=np.int32)
        self.check_like(func, np.float64)

    def test_like_dtype_str_kwarg(self):
        pyfunc = self.pyfunc

        def func(arr):
            return pyfunc(arr, dtype='int32')
        self.check_like(func, np.float64)

    def test_like_dtype_str_kwarg_alternative_spelling(self):
        pyfunc = self.pyfunc

        def func(arr):
            return pyfunc(arr, dtype='i4')
        self.check_like(func, np.float64)

    def test_like_dtype_non_const_str(self):
        pyfunc = self.pyfunc

        @njit
        def func(n, dt):
            return pyfunc(n, dt)
        with self.assertRaises(TypingError) as raises:
            func(np.ones(4), 'int32')
        excstr = str(raises.exception)
        msg = f'If np.{self.pyfunc.__name__} dtype is a string it must be a string constant.'
        self.assertIn(msg, excstr)
        self.assertIn('{}(array(float64, 1d, C), unicode_type)'.format(pyfunc.__name__), excstr)

    def test_like_dtype_invalid_str(self):
        pyfunc = self.pyfunc

        @njit
        def func(n):
            return pyfunc(n, 'ABCDEF')
        with self.assertRaises(TypingError) as raises:
            func(np.ones(4))
        excstr = str(raises.exception)
        self.assertIn("Invalid NumPy dtype specified: 'ABCDEF'", excstr)