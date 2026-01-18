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
class MatmulCommon:
    """Common tests for '@' operator and numpy.matmul.

    """
    types = '?bhilqBHILQefdgFDGO'

    def test_exceptions(self):
        dims = [((1,), (2,)), ((2, 1), (2,)), ((2,), (1, 2)), ((1, 2), (3, 1)), ((1,), ()), ((), 1), ((1, 1), ()), ((), (1, 1)), ((2, 2, 1), (3, 1, 2))]
        for dt, (dm1, dm2) in itertools.product(self.types, dims):
            a = np.ones(dm1, dtype=dt)
            b = np.ones(dm2, dtype=dt)
            assert_raises(ValueError, self.matmul, a, b)

    def test_shapes(self):
        dims = [((1, 1), (2, 1, 1)), ((2, 1, 1), (1, 1)), ((2, 1, 1), (2, 1, 1))]
        for dt, (dm1, dm2) in itertools.product(self.types, dims):
            a = np.ones(dm1, dtype=dt)
            b = np.ones(dm2, dtype=dt)
            res = self.matmul(a, b)
            assert_(res.shape == (2, 1, 1))
        for dt in self.types:
            a = np.ones((2,), dtype=dt)
            b = np.ones((2,), dtype=dt)
            c = self.matmul(a, b)
            assert_(np.array(c).shape == ())

    def test_result_types(self):
        mat = np.ones((1, 1))
        vec = np.ones((1,))
        for dt in self.types:
            m = mat.astype(dt)
            v = vec.astype(dt)
            for arg in [(m, v), (v, m), (m, m)]:
                res = self.matmul(*arg)
                assert_(res.dtype == dt)
            if dt != 'O':
                res = self.matmul(v, v)
                assert_(type(res) is np.dtype(dt).type)

    def test_scalar_output(self):
        vec1 = np.array([2])
        vec2 = np.array([3, 4]).reshape(1, -1)
        tgt = np.array([6, 8])
        for dt in self.types[1:]:
            v1 = vec1.astype(dt)
            v2 = vec2.astype(dt)
            res = self.matmul(v1, v2)
            assert_equal(res, tgt)
            res = self.matmul(v2.T, v1)
            assert_equal(res, tgt)
        vec = np.array([True, True], dtype='?').reshape(1, -1)
        res = self.matmul(vec[:, 0], vec)
        assert_equal(res, True)

    def test_vector_vector_values(self):
        vec1 = np.array([1, 2])
        vec2 = np.array([3, 4]).reshape(-1, 1)
        tgt1 = np.array([11])
        tgt2 = np.array([[3, 6], [4, 8]])
        for dt in self.types[1:]:
            v1 = vec1.astype(dt)
            v2 = vec2.astype(dt)
            res = self.matmul(v1, v2)
            assert_equal(res, tgt1)
            res = self.matmul(v2, v1.reshape(1, -1))
            assert_equal(res, tgt2)
        vec = np.array([True, True], dtype='?')
        res = self.matmul(vec, vec)
        assert_equal(res, True)

    def test_vector_matrix_values(self):
        vec = np.array([1, 2])
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.stack([mat1] * 2, axis=0)
        tgt1 = np.array([7, 10])
        tgt2 = np.stack([tgt1] * 2, axis=0)
        for dt in self.types[1:]:
            v = vec.astype(dt)
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            res = self.matmul(v, m1)
            assert_equal(res, tgt1)
            res = self.matmul(v, m2)
            assert_equal(res, tgt2)
        vec = np.array([True, False])
        mat1 = np.array([[True, False], [False, True]])
        mat2 = np.stack([mat1] * 2, axis=0)
        tgt1 = np.array([True, False])
        tgt2 = np.stack([tgt1] * 2, axis=0)
        res = self.matmul(vec, mat1)
        assert_equal(res, tgt1)
        res = self.matmul(vec, mat2)
        assert_equal(res, tgt2)

    def test_matrix_vector_values(self):
        vec = np.array([1, 2])
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.stack([mat1] * 2, axis=0)
        tgt1 = np.array([5, 11])
        tgt2 = np.stack([tgt1] * 2, axis=0)
        for dt in self.types[1:]:
            v = vec.astype(dt)
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            res = self.matmul(m1, v)
            assert_equal(res, tgt1)
            res = self.matmul(m2, v)
            assert_equal(res, tgt2)
        vec = np.array([True, False])
        mat1 = np.array([[True, False], [False, True]])
        mat2 = np.stack([mat1] * 2, axis=0)
        tgt1 = np.array([True, False])
        tgt2 = np.stack([tgt1] * 2, axis=0)
        res = self.matmul(vec, mat1)
        assert_equal(res, tgt1)
        res = self.matmul(vec, mat2)
        assert_equal(res, tgt2)

    def test_matrix_matrix_values(self):
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.array([[1, 0], [1, 1]])
        mat12 = np.stack([mat1, mat2], axis=0)
        mat21 = np.stack([mat2, mat1], axis=0)
        tgt11 = np.array([[7, 10], [15, 22]])
        tgt12 = np.array([[3, 2], [7, 4]])
        tgt21 = np.array([[1, 2], [4, 6]])
        tgt12_21 = np.stack([tgt12, tgt21], axis=0)
        tgt11_12 = np.stack((tgt11, tgt12), axis=0)
        tgt11_21 = np.stack((tgt11, tgt21), axis=0)
        for dt in self.types[1:]:
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            m12 = mat12.astype(dt)
            m21 = mat21.astype(dt)
            res = self.matmul(m1, m2)
            assert_equal(res, tgt12)
            res = self.matmul(m2, m1)
            assert_equal(res, tgt21)
            res = self.matmul(m12, m1)
            assert_equal(res, tgt11_21)
            res = self.matmul(m1, m12)
            assert_equal(res, tgt11_12)
            res = self.matmul(m12, m21)
            assert_equal(res, tgt12_21)
        m1 = np.array([[1, 1], [0, 0]], dtype=np.bool_)
        m2 = np.array([[1, 0], [1, 1]], dtype=np.bool_)
        m12 = np.stack([m1, m2], axis=0)
        m21 = np.stack([m2, m1], axis=0)
        tgt11 = m1
        tgt12 = m1
        tgt21 = np.array([[1, 1], [1, 1]], dtype=np.bool_)
        tgt12_21 = np.stack([tgt12, tgt21], axis=0)
        tgt11_12 = np.stack((tgt11, tgt12), axis=0)
        tgt11_21 = np.stack((tgt11, tgt21), axis=0)
        res = self.matmul(m1, m2)
        assert_equal(res, tgt12)
        res = self.matmul(m2, m1)
        assert_equal(res, tgt21)
        res = self.matmul(m12, m1)
        assert_equal(res, tgt11_21)
        res = self.matmul(m1, m12)
        assert_equal(res, tgt11_12)
        res = self.matmul(m12, m21)
        assert_equal(res, tgt12_21)