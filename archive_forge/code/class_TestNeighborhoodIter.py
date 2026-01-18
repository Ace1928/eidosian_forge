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
@pytest.mark.parametrize('dt', [float, Decimal], ids=['float', 'object'])
class TestNeighborhoodIter:

    def test_simple2d(self, dt):
        x = np.array([[0, 1], [2, 3]], dtype=dt)
        r = [np.array([[0, 0, 0], [0, 0, 1]], dtype=dt), np.array([[0, 0, 0], [0, 1, 0]], dtype=dt), np.array([[0, 0, 1], [0, 2, 3]], dtype=dt), np.array([[0, 1, 0], [2, 3, 0]], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator(x, [-1, 0, -1, 1], x[0], NEIGH_MODE['zero'])
        assert_array_equal(l, r)
        r = [np.array([[1, 1, 1], [1, 0, 1]], dtype=dt), np.array([[1, 1, 1], [0, 1, 1]], dtype=dt), np.array([[1, 0, 1], [1, 2, 3]], dtype=dt), np.array([[0, 1, 1], [2, 3, 1]], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator(x, [-1, 0, -1, 1], x[0], NEIGH_MODE['one'])
        assert_array_equal(l, r)
        r = [np.array([[4, 4, 4], [4, 0, 1]], dtype=dt), np.array([[4, 4, 4], [0, 1, 4]], dtype=dt), np.array([[4, 0, 1], [4, 2, 3]], dtype=dt), np.array([[0, 1, 4], [2, 3, 4]], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator(x, [-1, 0, -1, 1], 4, NEIGH_MODE['constant'])
        assert_array_equal(l, r)
        r = [np.array([[4, 0, 1], [4, 2, 3]], dtype=dt), np.array([[0, 1, 4], [2, 3, 4]], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator(x, [-1, 0, -1, 1], 4, NEIGH_MODE['constant'], 2)
        assert_array_equal(l, r)

    def test_mirror2d(self, dt):
        x = np.array([[0, 1], [2, 3]], dtype=dt)
        r = [np.array([[0, 0, 1], [0, 0, 1]], dtype=dt), np.array([[0, 1, 1], [0, 1, 1]], dtype=dt), np.array([[0, 0, 1], [2, 2, 3]], dtype=dt), np.array([[0, 1, 1], [2, 3, 3]], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator(x, [-1, 0, -1, 1], x[0], NEIGH_MODE['mirror'])
        assert_array_equal(l, r)

    def test_simple(self, dt):
        x = np.linspace(1, 5, 5).astype(dt)
        r = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 0]]
        l = _multiarray_tests.test_neighborhood_iterator(x, [-1, 1], x[0], NEIGH_MODE['zero'])
        assert_array_equal(l, r)
        r = [[1, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 1]]
        l = _multiarray_tests.test_neighborhood_iterator(x, [-1, 1], x[0], NEIGH_MODE['one'])
        assert_array_equal(l, r)
        r = [[x[4], 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, x[4]]]
        l = _multiarray_tests.test_neighborhood_iterator(x, [-1, 1], x[4], NEIGH_MODE['constant'])
        assert_array_equal(l, r)

    def test_mirror(self, dt):
        x = np.linspace(1, 5, 5).astype(dt)
        r = np.array([[2, 1, 1, 2, 3], [1, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 5], [3, 4, 5, 5, 4]], dtype=dt)
        l = _multiarray_tests.test_neighborhood_iterator(x, [-2, 2], x[1], NEIGH_MODE['mirror'])
        assert_([i.dtype == dt for i in l])
        assert_array_equal(l, r)

    def test_circular(self, dt):
        x = np.linspace(1, 5, 5).astype(dt)
        r = np.array([[4, 5, 1, 2, 3], [5, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 1], [3, 4, 5, 1, 2]], dtype=dt)
        l = _multiarray_tests.test_neighborhood_iterator(x, [-2, 2], x[0], NEIGH_MODE['circular'])
        assert_array_equal(l, r)