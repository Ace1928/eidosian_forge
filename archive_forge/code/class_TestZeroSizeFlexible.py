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
class TestZeroSizeFlexible:

    @staticmethod
    def _zeros(shape, dtype=str):
        dtype = np.dtype(dtype)
        if dtype == np.void:
            return np.zeros(shape, dtype=(dtype, 0))
        dtype = np.dtype([('x', dtype, 0)])
        return np.zeros(shape, dtype=dtype)['x']

    def test_create(self):
        zs = self._zeros(10, bytes)
        assert_equal(zs.itemsize, 0)
        zs = self._zeros(10, np.void)
        assert_equal(zs.itemsize, 0)
        zs = self._zeros(10, str)
        assert_equal(zs.itemsize, 0)

    def _test_sort_partition(self, name, kinds, **kwargs):
        for dt in [bytes, np.void, str]:
            zs = self._zeros(10, dt)
            sort_method = getattr(zs, name)
            sort_func = getattr(np, name)
            for kind in kinds:
                sort_method(kind=kind, **kwargs)
                sort_func(zs, kind=kind, **kwargs)

    def test_sort(self):
        self._test_sort_partition('sort', kinds='qhs')

    def test_argsort(self):
        self._test_sort_partition('argsort', kinds='qhs')

    def test_partition(self):
        self._test_sort_partition('partition', kinds=['introselect'], kth=2)

    def test_argpartition(self):
        self._test_sort_partition('argpartition', kinds=['introselect'], kth=2)

    def test_resize(self):
        for dt in [bytes, np.void, str]:
            zs = self._zeros(10, dt)
            zs.resize(25)
            zs.resize((10, 10))

    def test_view(self):
        for dt in [bytes, np.void, str]:
            zs = self._zeros(10, dt)
            assert_equal(zs.view(dt).dtype, np.dtype(dt))
            assert_equal(zs.view((dt, 1)).shape, (0,))

    def test_dumps(self):
        zs = self._zeros(10, int)
        assert_equal(zs, pickle.loads(zs.dumps()))

    def test_pickle(self):
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            for dt in [bytes, np.void, str]:
                zs = self._zeros(10, dt)
                p = pickle.dumps(zs, protocol=proto)
                zs2 = pickle.loads(p)
                assert_equal(zs.dtype, zs2.dtype)

    def test_pickle_empty(self):
        """Checking if an empty array pickled and un-pickled will not cause a
        segmentation fault"""
        arr = np.array([]).reshape(999999, 0)
        pk_dmp = pickle.dumps(arr)
        pk_load = pickle.loads(pk_dmp)
        assert pk_load.size == 0

    @pytest.mark.skipif(pickle.HIGHEST_PROTOCOL < 5, reason='requires pickle protocol 5')
    def test_pickle_with_buffercallback(self):
        array = np.arange(10)
        buffers = []
        bytes_string = pickle.dumps(array, buffer_callback=buffers.append, protocol=5)
        array_from_buffer = pickle.loads(bytes_string, buffers=buffers)
        array[0] = -1
        assert array_from_buffer[0] == -1, array_from_buffer[0]