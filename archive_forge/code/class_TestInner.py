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
class TestInner:

    def test_inner_type_mismatch(self):
        c = 1.0
        A = np.array((1, 1), dtype='i,i')
        assert_raises(TypeError, np.inner, c, A)
        assert_raises(TypeError, np.inner, A, c)

    def test_inner_scalar_and_vector(self):
        for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
            sca = np.array(3, dtype=dt)[()]
            vec = np.array([1, 2], dtype=dt)
            desired = np.array([3, 6], dtype=dt)
            assert_equal(np.inner(vec, sca), desired)
            assert_equal(np.inner(sca, vec), desired)

    def test_vecself(self):
        a = np.zeros(shape=(1, 80), dtype=np.float64)
        p = np.inner(a, a)
        assert_almost_equal(p, 0, decimal=14)

    def test_inner_product_with_various_contiguities(self):
        for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
            A = np.array([[1, 2], [3, 4]], dtype=dt)
            B = np.array([[1, 3], [2, 4]], dtype=dt)
            C = np.array([1, 1], dtype=dt)
            desired = np.array([4, 6], dtype=dt)
            assert_equal(np.inner(A.T, C), desired)
            assert_equal(np.inner(C, A.T), desired)
            assert_equal(np.inner(B, C), desired)
            assert_equal(np.inner(C, B), desired)
            desired = np.array([[7, 10], [15, 22]], dtype=dt)
            assert_equal(np.inner(A, B), desired)
            desired = np.array([[5, 11], [11, 25]], dtype=dt)
            assert_equal(np.inner(A, A), desired)
            assert_equal(np.inner(A, A.copy()), desired)
            a = np.arange(5).astype(dt)
            b = a[::-1]
            desired = np.array(10, dtype=dt).item()
            assert_equal(np.inner(b, a), desired)

    def test_3d_tensor(self):
        for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
            a = np.arange(24).reshape(2, 3, 4).astype(dt)
            b = np.arange(24, 48).reshape(2, 3, 4).astype(dt)
            desired = np.array([[[[158, 182, 206], [230, 254, 278]], [[566, 654, 742], [830, 918, 1006]], [[974, 1126, 1278], [1430, 1582, 1734]]], [[[1382, 1598, 1814], [2030, 2246, 2462]], [[1790, 2070, 2350], [2630, 2910, 3190]], [[2198, 2542, 2886], [3230, 3574, 3918]]]]).astype(dt)
            assert_equal(np.inner(a, b), desired)
            assert_equal(np.inner(b, a).transpose(2, 3, 0, 1), desired)