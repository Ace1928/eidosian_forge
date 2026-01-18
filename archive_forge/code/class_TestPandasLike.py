import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
class TestPandasLike(TestCase):
    """
    Test implementing a pandas-like Index object.
    Also stresses most of the high-level API.
    """

    def test_index_len(self):
        i = Index(np.arange(3))
        cfunc = jit(nopython=True)(len_usecase)
        self.assertPreciseEqual(cfunc(i), 3)

    def test_index_getitem(self):
        i = Index(np.int32([42, 8, -5]))
        cfunc = jit(nopython=True)(getitem_usecase)
        self.assertPreciseEqual(cfunc(i, 1), 8)
        ii = cfunc(i, slice(1, None))
        self.assertIsInstance(ii, Index)
        self.assertEqual(list(ii), [8, -5])

    def test_index_ufunc(self):
        """
        Check Numpy ufunc on an Index object.
        """
        i = Index(np.int32([42, 8, -5]))
        cfunc = jit(nopython=True)(npyufunc_usecase)
        ii = cfunc(i)
        self.assertIsInstance(ii, Index)
        self.assertPreciseEqual(ii._data, np.cos(np.sin(i._data)))

    def test_index_get_data(self):
        i = Index(np.int32([42, 8, -5]))
        cfunc = jit(nopython=True)(get_data_usecase)
        data = cfunc(i)
        self.assertIs(data, i._data)

    def test_index_is_monotonic(self):
        cfunc = jit(nopython=True)(is_monotonic_usecase)
        for values, expected in [([8, 42, 5], False), ([5, 8, 42], True), ([], True)]:
            i = Index(np.int32(values))
            got = cfunc(i)
            self.assertEqual(got, expected)

    def test_series_len(self):
        i = Index(np.int32([2, 4, 3]))
        s = Series(np.float64([1.5, 4.0, 2.5]), i)
        cfunc = jit(nopython=True)(len_usecase)
        self.assertPreciseEqual(cfunc(s), 3)

    def test_series_get_index(self):
        i = Index(np.int32([2, 4, 3]))
        s = Series(np.float64([1.5, 4.0, 2.5]), i)
        cfunc = jit(nopython=True)(get_index_usecase)
        got = cfunc(s)
        self.assertIsInstance(got, Index)
        self.assertIs(got._data, i._data)

    def test_series_ufunc(self):
        """
        Check Numpy ufunc on an Series object.
        """
        i = Index(np.int32([42, 8, -5]))
        s = Series(np.int64([1, 2, 3]), i)
        cfunc = jit(nopython=True)(npyufunc_usecase)
        ss = cfunc(s)
        self.assertIsInstance(ss, Series)
        self.assertIsInstance(ss._index, Index)
        self.assertIs(ss._index._data, i._data)
        self.assertPreciseEqual(ss._values, np.cos(np.sin(s._values)))

    def test_series_constructor(self):
        i = Index(np.int32([42, 8, -5]))
        d = np.float64([1.5, 4.0, 2.5])
        cfunc = jit(nopython=True)(make_series_usecase)
        got = cfunc(d, i)
        self.assertIsInstance(got, Series)
        self.assertIsInstance(got._index, Index)
        self.assertIs(got._index._data, i._data)
        self.assertIs(got._values, d)

    def test_series_clip(self):
        i = Index(np.int32([42, 8, -5]))
        s = Series(np.float64([1.5, 4.0, 2.5]), i)
        cfunc = jit(nopython=True)(clip_usecase)
        ss = cfunc(s, 1.6, 3.0)
        self.assertIsInstance(ss, Series)
        self.assertIsInstance(ss._index, Index)
        self.assertIs(ss._index._data, i._data)
        self.assertPreciseEqual(ss._values, np.float64([1.6, 3.0, 2.5]))