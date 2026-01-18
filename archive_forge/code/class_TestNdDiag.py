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
class TestNdDiag(TestCase):

    def setUp(self):
        v = np.array([1, 2, 3])
        hv = np.array([[1, 2, 3]])
        vv = np.transpose(hv)
        self.vectors = [v, hv, vv]
        a3x4 = np.arange(12).reshape(3, 4)
        a4x3 = np.arange(12).reshape(4, 3)
        self.matricies = [a3x4, a4x3]

        def func(q):
            return np.diag(q)
        self.py = func
        self.jit = nrtjit(func)

        def func_kwarg(q, k=0):
            return np.diag(q, k=k)
        self.py_kw = func_kwarg
        self.jit_kw = nrtjit(func_kwarg)

    def check_diag(self, pyfunc, nrtfunc, *args, **kwargs):
        expected = pyfunc(*args, **kwargs)
        computed = nrtfunc(*args, **kwargs)
        self.assertEqual(computed.size, expected.size)
        self.assertEqual(computed.dtype, expected.dtype)
        np.testing.assert_equal(expected, computed)

    def test_diag_vect_create(self):
        for d in self.vectors:
            self.check_diag(self.py, self.jit, d)

    def test_diag_vect_create_kwarg(self):
        for k in range(-10, 10):
            for d in self.vectors:
                self.check_diag(self.py_kw, self.jit_kw, d, k=k)

    def test_diag_extract(self):
        for d in self.matricies:
            self.check_diag(self.py, self.jit, d)

    def test_diag_extract_kwarg(self):
        for k in range(-4, 4):
            for d in self.matricies:
                self.check_diag(self.py_kw, self.jit_kw, d, k=k)

    def test_error_handling(self):
        d = np.array([[[1.0]]])
        cfunc = nrtjit(self.py)
        with self.assertRaises(TypeError):
            cfunc()
        with self.assertRaises(TypingError):
            cfunc(d)
        with self.assertRaises(TypingError):
            dfunc = nrtjit(self.py_kw)
            dfunc(d, k=3)

    def test_bad_shape(self):
        cfunc = nrtjit(self.py)
        msg = '.*The argument "v" must be array-like.*'
        with self.assertRaisesRegex(TypingError, msg) as raises:
            cfunc(None)