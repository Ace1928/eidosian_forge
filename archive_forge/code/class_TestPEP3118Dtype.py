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
class TestPEP3118Dtype:

    def _check(self, spec, wanted):
        dt = np.dtype(wanted)
        actual = _dtype_from_pep3118(spec)
        assert_equal(actual, dt, err_msg='spec %r != dtype %r' % (spec, wanted))

    def test_native_padding(self):
        align = np.dtype('i').alignment
        for j in range(8):
            if j == 0:
                s = 'bi'
            else:
                s = 'b%dxi' % j
            self._check('@' + s, {'f0': ('i1', 0), 'f1': ('i', align * (1 + j // align))})
            self._check('=' + s, {'f0': ('i1', 0), 'f1': ('i', 1 + j)})

    def test_native_padding_2(self):
        self._check('x3T{xi}', {'f0': (({'f0': ('i', 4)}, (3,)), 4)})
        self._check('^x3T{xi}', {'f0': (({'f0': ('i', 1)}, (3,)), 1)})

    def test_trailing_padding(self):
        align = np.dtype('i').alignment
        size = np.dtype('i').itemsize

        def aligned(n):
            return align * (1 + (n - 1) // align)
        base = dict(formats=['i'], names=['f0'])
        self._check('ix', dict(itemsize=aligned(size + 1), **base))
        self._check('ixx', dict(itemsize=aligned(size + 2), **base))
        self._check('ixxx', dict(itemsize=aligned(size + 3), **base))
        self._check('ixxxx', dict(itemsize=aligned(size + 4), **base))
        self._check('i7x', dict(itemsize=aligned(size + 7), **base))
        self._check('^ix', dict(itemsize=size + 1, **base))
        self._check('^ixx', dict(itemsize=size + 2, **base))
        self._check('^ixxx', dict(itemsize=size + 3, **base))
        self._check('^ixxxx', dict(itemsize=size + 4, **base))
        self._check('^i7x', dict(itemsize=size + 7, **base))

    def test_native_padding_3(self):
        dt = np.dtype([('a', 'b'), ('b', 'i'), ('sub', np.dtype('b,i')), ('c', 'i')], align=True)
        self._check('T{b:a:xxxi:b:T{b:f0:=i:f1:}:sub:xxxi:c:}', dt)
        dt = np.dtype([('a', 'b'), ('b', 'i'), ('c', 'b'), ('d', 'b'), ('e', 'b'), ('sub', np.dtype('b,i', align=True))])
        self._check('T{b:a:=i:b:b:c:b:d:b:e:T{b:f0:xxxi:f1:}:sub:}', dt)

    def test_padding_with_array_inside_struct(self):
        dt = np.dtype([('a', 'b'), ('b', 'i'), ('c', 'b', (3,)), ('d', 'i')], align=True)
        self._check('T{b:a:xxxi:b:3b:c:xi:d:}', dt)

    def test_byteorder_inside_struct(self):
        self._check('@T{^i}xi', {'f0': ({'f0': ('i', 0)}, 0), 'f1': ('i', 5)})

    def test_intra_padding(self):
        align = np.dtype('i').alignment
        size = np.dtype('i').itemsize

        def aligned(n):
            return align * (1 + (n - 1) // align)
        self._check('(3)T{ix}', (dict(names=['f0'], formats=['i'], offsets=[0], itemsize=aligned(size + 1)), (3,)))

    def test_char_vs_string(self):
        dt = np.dtype('c')
        self._check('c', dt)
        dt = np.dtype([('f0', 'S1', (4,)), ('f1', 'S4')])
        self._check('4c4s', dt)

    def test_field_order(self):
        self._check('(0)I:a:f:b:', [('a', 'I', (0,)), ('b', 'f')])
        self._check('(0)I:b:f:a:', [('b', 'I', (0,)), ('a', 'f')])

    def test_unnamed_fields(self):
        self._check('ii', [('f0', 'i'), ('f1', 'i')])
        self._check('ii:f0:', [('f1', 'i'), ('f0', 'i')])
        self._check('i', 'i')
        self._check('i:f0:', [('f0', 'i')])