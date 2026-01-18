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
def _test_cast_from_flexible(self, dtype):
    for n in range(3):
        v = np.array(b'', (dtype, n))
        assert_equal(bool(v), False)
        assert_equal(bool(v[()]), False)
        assert_equal(v.astype(bool), False)
        assert_(isinstance(v.astype(bool), np.ndarray))
        assert_(v[()].astype(bool) is np.False_)
    for n in range(1, 4):
        for val in [b'a', b'0', b' ']:
            v = np.array(val, (dtype, n))
            assert_equal(bool(v), True)
            assert_equal(bool(v[()]), True)
            assert_equal(v.astype(bool), True)
            assert_(isinstance(v.astype(bool), np.ndarray))
            assert_(v[()].astype(bool) is np.True_)