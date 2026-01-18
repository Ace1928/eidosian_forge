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
class TestBytestringArrayNonzero:

    def test_empty_bstring_array_is_falsey(self):
        assert_(not np.array([''], dtype=str))

    def test_whitespace_bstring_array_is_falsey(self):
        a = np.array(['spam'], dtype=str)
        a[0] = '  \x00\x00'
        assert_(not a)

    def test_all_null_bstring_array_is_falsey(self):
        a = np.array(['spam'], dtype=str)
        a[0] = '\x00\x00\x00\x00'
        assert_(not a)

    def test_null_inside_bstring_array_is_truthy(self):
        a = np.array(['spam'], dtype=str)
        a[0] = ' \x00 \x00'
        assert_(a)