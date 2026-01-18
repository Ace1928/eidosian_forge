from __future__ import annotations
import collections.abc
import tempfile
import sys
import warnings
import operator
import io
import itertools
import functools
import ctypes
import os
import gc
import re
import weakref
import pytest
from contextlib import contextmanager
from numpy.compat import pickle
import pathlib
import builtins
from decimal import Decimal
import mmap
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.testing._private.utils import requires_memory, _no_tracing
from numpy.core.tests._locales import CommaDecimalPointLocale
from numpy.lib.recfunctions import repack_fields
from numpy.core.multiarray import _get_ndarray_c_version
from datetime import timedelta, datetime
from numpy.core._internal import _dtype_from_pep3118
from numpy.testing import IS_PYPY
class TestLexsort:

    @pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64])
    def test_basic(self, dtype):
        a = np.array([1, 2, 1, 3, 1, 5], dtype=dtype)
        b = np.array([0, 4, 5, 6, 2, 3], dtype=dtype)
        idx = np.lexsort((b, a))
        expected_idx = np.array([0, 4, 2, 1, 3, 5])
        assert_array_equal(idx, expected_idx)
        assert_array_equal(a[idx], np.sort(a))

    def test_mixed(self):
        a = np.array([1, 2, 1, 3, 1, 5])
        b = np.array([0, 4, 5, 6, 2, 3], dtype='datetime64[D]')
        idx = np.lexsort((b, a))
        expected_idx = np.array([0, 4, 2, 1, 3, 5])
        assert_array_equal(idx, expected_idx)

    def test_datetime(self):
        a = np.array([0, 0, 0], dtype='datetime64[D]')
        b = np.array([2, 1, 0], dtype='datetime64[D]')
        idx = np.lexsort((b, a))
        expected_idx = np.array([2, 1, 0])
        assert_array_equal(idx, expected_idx)
        a = np.array([0, 0, 0], dtype='timedelta64[D]')
        b = np.array([2, 1, 0], dtype='timedelta64[D]')
        idx = np.lexsort((b, a))
        expected_idx = np.array([2, 1, 0])
        assert_array_equal(idx, expected_idx)

    def test_object(self):
        a = np.random.choice(10, 1000)
        b = np.random.choice(['abc', 'xy', 'wz', 'efghi', 'qwst', 'x'], 1000)
        for u in (a, b):
            left = np.lexsort((u.astype('O'),))
            right = np.argsort(u, kind='mergesort')
            assert_array_equal(left, right)
        for u, v in ((a, b), (b, a)):
            idx = np.lexsort((u, v))
            assert_array_equal(idx, np.lexsort((u.astype('O'), v)))
            assert_array_equal(idx, np.lexsort((u, v.astype('O'))))
            u, v = (np.array(u, dtype='object'), np.array(v, dtype='object'))
            assert_array_equal(idx, np.lexsort((u, v)))

    def test_invalid_axis(self):
        x = np.linspace(0.0, 1.0, 42 * 3).reshape(42, 3)
        assert_raises(np.AxisError, np.lexsort, x, axis=2)