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
class TestZeroRank:

    def setup_method(self):
        self.d = (np.array(0), np.array('x', object))

    def test_ellipsis_subscript(self):
        a, b = self.d
        assert_equal(a[...], 0)
        assert_equal(b[...], 'x')
        assert_(a[...].base is a)
        assert_(b[...].base is b)

    def test_empty_subscript(self):
        a, b = self.d
        assert_equal(a[()], 0)
        assert_equal(b[()], 'x')
        assert_(type(a[()]) is a.dtype.type)
        assert_(type(b[()]) is str)

    def test_invalid_subscript(self):
        a, b = self.d
        assert_raises(IndexError, lambda x: x[0], a)
        assert_raises(IndexError, lambda x: x[0], b)
        assert_raises(IndexError, lambda x: x[np.array([], int)], a)
        assert_raises(IndexError, lambda x: x[np.array([], int)], b)

    def test_ellipsis_subscript_assignment(self):
        a, b = self.d
        a[...] = 42
        assert_equal(a, 42)
        b[...] = ''
        assert_equal(b.item(), '')

    def test_empty_subscript_assignment(self):
        a, b = self.d
        a[()] = 42
        assert_equal(a, 42)
        b[()] = ''
        assert_equal(b.item(), '')

    def test_invalid_subscript_assignment(self):
        a, b = self.d

        def assign(x, i, v):
            x[i] = v
        assert_raises(IndexError, assign, a, 0, 42)
        assert_raises(IndexError, assign, b, 0, '')
        assert_raises(ValueError, assign, a, (), '')

    def test_newaxis(self):
        a, b = self.d
        assert_equal(a[np.newaxis].shape, (1,))
        assert_equal(a[..., np.newaxis].shape, (1,))
        assert_equal(a[np.newaxis, ...].shape, (1,))
        assert_equal(a[..., np.newaxis].shape, (1,))
        assert_equal(a[np.newaxis, ..., np.newaxis].shape, (1, 1))
        assert_equal(a[..., np.newaxis, np.newaxis].shape, (1, 1))
        assert_equal(a[np.newaxis, np.newaxis, ...].shape, (1, 1))
        assert_equal(a[(np.newaxis,) * 10].shape, (1,) * 10)

    def test_invalid_newaxis(self):
        a, b = self.d

        def subscript(x, i):
            x[i]
        assert_raises(IndexError, subscript, a, (np.newaxis, 0))
        assert_raises(IndexError, subscript, a, (np.newaxis,) * 50)

    def test_constructor(self):
        x = np.ndarray(())
        x[()] = 5
        assert_equal(x[()], 5)
        y = np.ndarray((), buffer=x)
        y[()] = 6
        assert_equal(x[()], 6)
        with pytest.raises(ValueError):
            np.ndarray((2,), strides=())
        with pytest.raises(ValueError):
            np.ndarray((), strides=(2,))

    def test_output(self):
        x = np.array(2)
        assert_raises(ValueError, np.add, x, [1], x)

    def test_real_imag(self):
        x = np.array(1j)
        xr = x.real
        xi = x.imag
        assert_equal(xr, np.array(0))
        assert_(type(xr) is np.ndarray)
        assert_equal(xr.flags.contiguous, True)
        assert_equal(xr.flags.f_contiguous, True)
        assert_equal(xi, np.array(1))
        assert_(type(xi) is np.ndarray)
        assert_equal(xi.flags.contiguous, True)
        assert_equal(xi.flags.f_contiguous, True)