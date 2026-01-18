import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
class TestFmin(_FilterInvalids):

    def test_reduce(self):
        dflt = np.typecodes['AllFloat']
        dint = np.typecodes['AllInteger']
        seq1 = np.arange(11)
        seq2 = seq1[::-1]
        func = np.fmin.reduce
        for dt in dint:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 0)
            assert_equal(func(tmp2), 0)
        for dt in dflt:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 0)
            assert_equal(func(tmp2), 0)
            tmp1[::2] = np.nan
            tmp2[::2] = np.nan
            assert_equal(func(tmp1), 1)
            assert_equal(func(tmp2), 1)

    def test_reduce_complex(self):
        assert_equal(np.fmin.reduce([1, 2j]), 2j)
        assert_equal(np.fmin.reduce([1 + 3j, 2j]), 2j)

    def test_float_nans(self):
        nan = np.nan
        arg1 = np.array([0, nan, nan])
        arg2 = np.array([nan, 0, nan])
        out = np.array([0, 0, nan])
        assert_equal(np.fmin(arg1, arg2), out)

    def test_complex_nans(self):
        nan = np.nan
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = np.array([0, cnan, cnan], dtype=complex)
            arg2 = np.array([cnan, 0, cnan], dtype=complex)
            out = np.array([0, 0, nan], dtype=complex)
            assert_equal(np.fmin(arg1, arg2), out)

    def test_precision(self):
        dtypes = [np.float16, np.float32, np.float64, np.longdouble]
        for dt in dtypes:
            dtmin = np.finfo(dt).min
            dtmax = np.finfo(dt).max
            d1 = dt(0.1)
            d1_next = np.nextafter(d1, np.inf)
            test_cases = [(dtmin, np.inf, dtmin), (dtmax, np.inf, dtmax), (d1, d1_next, d1), (dtmin, np.nan, dtmin)]
            for v1, v2, expected in test_cases:
                assert_equal(np.fmin([v1], [v2]), [expected])
                assert_equal(np.fmin.reduce([v1, v2]), expected)