import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
class TestUFuncBadArgs(TestCase):

    def test_missing_args(self):

        def func(x):
            """error: np.add requires two args"""
            result = np.add(x)
            return result
        with self.assertRaises(TypingError):
            njit([types.float64(types.float64)])(func)

    def test_too_many_args(self):

        def func(x, out, out2):
            """error: too many args"""
            result = np.add(x, x, out, out2)
            return result
        array_type = types.Array(types.float64, 1, 'C')
        sig = array_type(array_type, array_type, array_type)
        with self.assertRaises(TypingError):
            njit(sig)(func)

    def test_no_scalar_result_by_reference(self):

        def func(x):
            """error: scalar as a return value is not supported"""
            y = 0
            np.add(x, x, y)
        with self.assertRaises(TypingError):
            njit([types.float64(types.float64)])(func)