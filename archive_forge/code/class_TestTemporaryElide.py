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
class TestTemporaryElide:

    def test_extension_incref_elide(self):
        from numpy.core._multiarray_tests import incref_elide
        d = np.ones(100000)
        orig, res = incref_elide(d)
        d + d
        assert_array_equal(orig, d)
        assert_array_equal(res, d + d)

    def test_extension_incref_elide_stack(self):
        from numpy.core._multiarray_tests import incref_elide_l
        l = [1, 1, 1, 1, np.ones(100000)]
        res = incref_elide_l(l)
        assert_array_equal(l[4], np.ones(100000))
        assert_array_equal(res, l[4] + l[4])

    def test_temporary_with_cast(self):
        d = np.ones(200000, dtype=np.int64)
        assert_equal((d + d + 2 ** 222).dtype, np.dtype('O'))
        r = (d + d) / 2
        assert_equal(r.dtype, np.dtype('f8'))
        r = np.true_divide(d + d, 2)
        assert_equal(r.dtype, np.dtype('f8'))
        r = (d + d) / 2.0
        assert_equal(r.dtype, np.dtype('f8'))
        r = (d + d) // 2
        assert_equal(r.dtype, np.dtype(np.int64))
        f = np.ones(100000, dtype=np.float32)
        assert_equal((f + f + f.astype(np.float64)).dtype, np.dtype('f8'))
        d = f.astype(np.float64)
        assert_equal((f + f + d).dtype, d.dtype)
        l = np.ones(100000, dtype=np.longdouble)
        assert_equal((d + d + l).dtype, l.dtype)
        for dt in (np.complex64, np.complex128, np.clongdouble):
            c = np.ones(100000, dtype=dt)
            r = abs(c * 2.0)
            assert_equal(r.dtype, np.dtype('f%d' % (c.itemsize // 2)))

    def test_elide_broadcast(self):
        d = np.ones((2000, 1), dtype=int)
        b = np.ones(2000, dtype=bool)
        r = 1 - d + b
        assert_equal(r, 1)
        assert_equal(r.shape, (2000, 2000))

    def test_elide_scalar(self):
        a = np.bool_()
        assert_(type(~(a & a)) is np.bool_)

    def test_elide_scalar_readonly(self):
        a = np.empty(100000, dtype=np.float64)
        a.imag ** 2

    def test_elide_readonly(self):
        r = np.asarray(np.broadcast_to(np.zeros(1), 100000).flat) * 0.0
        assert_equal(r, 0)

    def test_elide_updateifcopy(self):
        a = np.ones(2 ** 20)[::2]
        b = a.flat.__array__() + 1
        del b
        assert_equal(a, 1)