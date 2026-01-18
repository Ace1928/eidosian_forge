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
class TestAssignment:

    def test_assignment_broadcasting(self):
        a = np.arange(6).reshape(2, 3)
        a[...] = np.arange(3)
        assert_equal(a, [[0, 1, 2], [0, 1, 2]])
        a[...] = np.arange(2).reshape(2, 1)
        assert_equal(a, [[0, 0, 0], [1, 1, 1]])
        a[...] = np.arange(6)[::-1].reshape(1, 2, 3)
        assert_equal(a, [[5, 4, 3], [2, 1, 0]])

        def assign(a, b):
            a[...] = b
        assert_raises(ValueError, assign, a, np.arange(12).reshape(2, 2, 3))

    def test_assignment_errors(self):

        class C:
            pass
        a = np.zeros(1)

        def assign(v):
            a[0] = v
        assert_raises((AttributeError, TypeError), assign, C())
        assert_raises(ValueError, assign, [1])

    def test_unicode_assignment(self):
        from numpy.core.numeric import set_string_function

        @contextmanager
        def inject_str(s):
            """ replace ndarray.__str__ temporarily """
            set_string_function(lambda x: s, repr=False)
            try:
                yield
            finally:
                set_string_function(None, repr=False)
        a1d = np.array(['test'])
        a0d = np.array('done')
        with inject_str('bad'):
            a1d[0] = a0d
        assert_equal(a1d[0], 'done')
        np.array([np.array('åäö')])

    def test_stringlike_empty_list(self):
        u = np.array(['done'])
        b = np.array([b'done'])

        class bad_sequence:

            def __getitem__(self):
                pass

            def __len__(self):
                raise RuntimeError
        assert_raises(ValueError, operator.setitem, u, 0, [])
        assert_raises(ValueError, operator.setitem, b, 0, [])
        assert_raises(ValueError, operator.setitem, u, 0, bad_sequence())
        assert_raises(ValueError, operator.setitem, b, 0, bad_sequence())

    def test_longdouble_assignment(self):
        for dtype in (np.longdouble, np.longcomplex):
            tinyb = np.nextafter(np.longdouble(0), 1).astype(dtype)
            tinya = np.nextafter(np.longdouble(0), -1).astype(dtype)
            tiny1d = np.array([tinya])
            assert_equal(tiny1d[0], tinya)
            tiny1d[0] = tinyb
            assert_equal(tiny1d[0], tinyb)
            tiny1d[0, ...] = tinya
            assert_equal(tiny1d[0], tinya)
            tiny1d[0, ...] = tinyb[...]
            assert_equal(tiny1d[0], tinyb)
            tiny1d[0] = tinyb[...]
            assert_equal(tiny1d[0], tinyb)
            arr = np.array([np.array(tinya)])
            assert_equal(arr[0], tinya)

    def test_cast_to_string(self):
        a = np.zeros(1, dtype='S20')
        a[:] = np.array(['1.12345678901234567890'], dtype='f8')
        assert_equal(a[0], b'1.1234567890123457')