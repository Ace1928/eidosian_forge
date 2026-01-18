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
class TestArgmax:
    usg_data = [([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 0), ([3, 3, 3, 3, 2, 2, 2, 2], 0), ([0, 1, 2, 3, 4, 5, 6, 7], 7), ([7, 6, 5, 4, 3, 2, 1, 0], 0)]
    sg_data = usg_data + [([1, 2, 3, 4, -4, -3, -2, -1], 3), ([1, 2, 3, 4, -1, -2, -3, -4], 3)]
    darr = [(np.array(d[0], dtype=t), d[1]) for d, t in itertools.product(usg_data, (np.uint8, np.uint16, np.uint32, np.uint64))]
    darr = darr + [(np.array(d[0], dtype=t), d[1]) for d, t in itertools.product(sg_data, (np.int8, np.int16, np.int32, np.int64, np.float32, np.float64))]
    darr = darr + [(np.array(d[0], dtype=t), d[1]) for d, t in itertools.product((([0, 1, 2, 3, np.nan], 4), ([0, 1, 2, np.nan, 3], 3), ([np.nan, 0, 1, 2, 3], 0), ([np.nan, 0, np.nan, 2, 3], 0), ([1] * (2 * 5 - 1) + [np.nan], 2 * 5 - 1), ([1] * (4 * 5 - 1) + [np.nan], 4 * 5 - 1), ([1] * (8 * 5 - 1) + [np.nan], 8 * 5 - 1), ([1] * (16 * 5 - 1) + [np.nan], 16 * 5 - 1), ([1] * (32 * 5 - 1) + [np.nan], 32 * 5 - 1)), (np.float32, np.float64))]
    nan_arr = darr + [([0, 1, 2, 3, complex(0, np.nan)], 4), ([0, 1, 2, 3, complex(np.nan, 0)], 4), ([0, 1, 2, complex(np.nan, 0), 3], 3), ([0, 1, 2, complex(0, np.nan), 3], 3), ([complex(0, np.nan), 0, 1, 2, 3], 0), ([complex(np.nan, np.nan), 0, 1, 2, 3], 0), ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, 1)], 0), ([complex(np.nan, np.nan), complex(np.nan, 2), complex(np.nan, 1)], 0), ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, np.nan)], 0), ([complex(0, 0), complex(0, 2), complex(0, 1)], 1), ([complex(1, 0), complex(0, 2), complex(0, 1)], 0), ([complex(1, 0), complex(0, 2), complex(1, 1)], 2), ([np.datetime64('1923-04-14T12:43:12'), np.datetime64('1994-06-21T14:43:15'), np.datetime64('2001-10-15T04:10:32'), np.datetime64('1995-11-25T16:02:16'), np.datetime64('2005-01-04T03:14:12'), np.datetime64('2041-12-03T14:05:03')], 5), ([np.datetime64('1935-09-14T04:40:11'), np.datetime64('1949-10-12T12:32:11'), np.datetime64('2010-01-03T05:14:12'), np.datetime64('2015-11-20T12:20:59'), np.datetime64('1932-09-23T10:10:13'), np.datetime64('2014-10-10T03:50:30')], 3), ([np.datetime64('NaT'), np.datetime64('NaT'), np.datetime64('2010-01-03T05:14:12'), np.datetime64('NaT'), np.datetime64('2015-09-23T10:10:13'), np.datetime64('1932-10-10T03:50:30')], 0), ([np.datetime64('2059-03-14T12:43:12'), np.datetime64('1996-09-21T14:43:15'), np.datetime64('NaT'), np.datetime64('2022-12-25T16:02:16'), np.datetime64('1963-10-04T03:14:12'), np.datetime64('2013-05-08T18:15:23')], 2), ([np.timedelta64(2, 's'), np.timedelta64(1, 's'), np.timedelta64('NaT', 's'), np.timedelta64(3, 's')], 2), ([np.timedelta64('NaT', 's')] * 3, 0), ([timedelta(days=5, seconds=14), timedelta(days=2, seconds=35), timedelta(days=-1, seconds=23)], 0), ([timedelta(days=1, seconds=43), timedelta(days=10, seconds=5), timedelta(days=5, seconds=14)], 1), ([timedelta(days=10, seconds=24), timedelta(days=10, seconds=5), timedelta(days=10, seconds=43)], 2), ([False, False, False, False, True], 4), ([False, False, False, True, False], 3), ([True, False, False, False, False], 0), ([True, False, True, False, False], 0)]

    @pytest.mark.parametrize('data', nan_arr)
    def test_combinations(self, data):
        arr, pos = data
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in reduce')
            val = np.max(arr)
        assert_equal(np.argmax(arr), pos, err_msg='%r' % arr)
        assert_equal(arr[np.argmax(arr)], val, err_msg='%r' % arr)
        rarr = np.repeat(arr, 129)
        rpos = pos * 129
        assert_equal(np.argmax(rarr), rpos, err_msg='%r' % rarr)
        assert_equal(rarr[np.argmax(rarr)], val, err_msg='%r' % rarr)
        padd = np.repeat(np.min(arr), 513)
        rarr = np.concatenate((arr, padd))
        rpos = pos
        assert_equal(np.argmax(rarr), rpos, err_msg='%r' % rarr)
        assert_equal(rarr[np.argmax(rarr)], val, err_msg='%r' % rarr)

    def test_maximum_signed_integers(self):
        a = np.array([1, 2 ** 7 - 1, -2 ** 7], dtype=np.int8)
        assert_equal(np.argmax(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)
        a = np.array([1, 2 ** 15 - 1, -2 ** 15], dtype=np.int16)
        assert_equal(np.argmax(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)
        a = np.array([1, 2 ** 31 - 1, -2 ** 31], dtype=np.int32)
        assert_equal(np.argmax(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)
        a = np.array([1, 2 ** 63 - 1, -2 ** 63], dtype=np.int64)
        assert_equal(np.argmax(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)