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
class TestPutmask:

    def tst_basic(self, x, T, mask, val):
        np.putmask(x, mask, val)
        assert_equal(x[mask], np.array(val, T))

    def test_ip_types(self):
        unchecked_types = [bytes, str, np.void]
        x = np.random.random(1000) * 100
        mask = x < 40
        for val in [-100, 0, 15]:
            for types in np.sctypes.values():
                for T in types:
                    if T not in unchecked_types:
                        if val < 0 and np.dtype(T).kind == 'u':
                            val = np.iinfo(T).max - 99
                        self.tst_basic(x.copy().astype(T), T, mask, val)
            dt = np.dtype('S3')
            self.tst_basic(x.astype(dt), dt.type, mask, dt.type(val)[:3])

    def test_mask_size(self):
        assert_raises(ValueError, np.putmask, np.array([1, 2, 3]), [True], 5)

    @pytest.mark.parametrize('dtype', ('>i4', '<i4'))
    def test_byteorder(self, dtype):
        x = np.array([1, 2, 3], dtype)
        np.putmask(x, [True, False, True], -1)
        assert_array_equal(x, [-1, 2, -1])

    def test_record_array(self):
        rec = np.array([(-5, 2.0, 3.0), (5.0, 4.0, 3.0)], dtype=[('x', '<f8'), ('y', '>f8'), ('z', '<f8')])
        np.putmask(rec['x'], [True, False], 10)
        assert_array_equal(rec['x'], [10, 5])
        assert_array_equal(rec['y'], [2, 4])
        assert_array_equal(rec['z'], [3, 3])
        np.putmask(rec['y'], [True, False], 11)
        assert_array_equal(rec['x'], [10, 5])
        assert_array_equal(rec['y'], [11, 4])
        assert_array_equal(rec['z'], [3, 3])

    def test_overlaps(self):
        x = np.array([True, False, True, False])
        np.putmask(x[1:4], [True, True, True], x[:3])
        assert_equal(x, np.array([True, True, False, True]))
        x = np.array([True, False, True, False])
        np.putmask(x[1:4], x[:3], [True, False, True])
        assert_equal(x, np.array([True, True, True, True]))

    def test_writeable(self):
        a = np.arange(5)
        a.flags.writeable = False
        with pytest.raises(ValueError):
            np.putmask(a, a >= 2, 3)

    def test_kwargs(self):
        x = np.array([0, 0])
        np.putmask(x, [0, 1], [-1, -2])
        assert_array_equal(x, [0, -2])
        x = np.array([0, 0])
        np.putmask(x, mask=[0, 1], values=[-1, -2])
        assert_array_equal(x, [0, -2])
        x = np.array([0, 0])
        np.putmask(x, values=[-1, -2], mask=[0, 1])
        assert_array_equal(x, [0, -2])
        with pytest.raises(TypeError):
            np.putmask(a=x, values=[-1, -2], mask=[0, 1])