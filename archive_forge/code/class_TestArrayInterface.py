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
class TestArrayInterface:

    class Foo:

        def __init__(self, value):
            self.value = value
            self.iface = {'typestr': 'f8'}

        def __float__(self):
            return float(self.value)

        @property
        def __array_interface__(self):
            return self.iface
    f = Foo(0.5)

    @pytest.mark.parametrize('val, iface, expected', [(f, {}, 0.5), ([f], {}, [0.5]), ([f, f], {}, [0.5, 0.5]), (f, {'shape': ()}, 0.5), (f, {'shape': None}, TypeError), (f, {'shape': (1, 1)}, [[0.5]]), (f, {'shape': (2,)}, ValueError), (f, {'strides': ()}, 0.5), (f, {'strides': (2,)}, ValueError), (f, {'strides': 16}, TypeError)])
    def test_scalar_interface(self, val, iface, expected):
        self.f.iface = {'typestr': 'f8'}
        self.f.iface.update(iface)
        if HAS_REFCOUNT:
            pre_cnt = sys.getrefcount(np.dtype('f8'))
        if isinstance(expected, type):
            assert_raises(expected, np.array, val)
        else:
            result = np.array(val)
            assert_equal(np.array(val), expected)
            assert result.dtype == 'f8'
            del result
        if HAS_REFCOUNT:
            post_cnt = sys.getrefcount(np.dtype('f8'))
            assert_equal(pre_cnt, post_cnt)