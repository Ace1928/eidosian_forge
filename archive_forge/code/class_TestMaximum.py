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
class TestMaximum(_FilterInvalids):

    def test_reduce(self):
        dflt = np.typecodes['AllFloat']
        dint = np.typecodes['AllInteger']
        seq1 = np.arange(11)
        seq2 = seq1[::-1]
        func = np.maximum.reduce
        for dt in dint:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 10)
            assert_equal(func(tmp2), 10)
        for dt in dflt:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 10)
            assert_equal(func(tmp2), 10)
            tmp1[::2] = np.nan
            tmp2[::2] = np.nan
            assert_equal(func(tmp1), np.nan)
            assert_equal(func(tmp2), np.nan)

    def test_reduce_complex(self):
        assert_equal(np.maximum.reduce([1, 2j]), 1)
        assert_equal(np.maximum.reduce([1 + 3j, 2j]), 1 + 3j)

    def test_float_nans(self):
        nan = np.nan
        arg1 = np.array([0, nan, nan])
        arg2 = np.array([nan, 0, nan])
        out = np.array([nan, nan, nan])
        assert_equal(np.maximum(arg1, arg2), out)

    def test_object_nans(self):
        for i in range(1):
            x = np.array(float('nan'), object)
            y = 1.0
            z = np.array(float('nan'), object)
            assert_(np.maximum(x, y) == 1.0)
            assert_(np.maximum(z, y) == 1.0)

    def test_complex_nans(self):
        nan = np.nan
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = np.array([0, cnan, cnan], dtype=complex)
            arg2 = np.array([cnan, 0, cnan], dtype=complex)
            out = np.array([nan, nan, nan], dtype=complex)
            assert_equal(np.maximum(arg1, arg2), out)

    def test_object_array(self):
        arg1 = np.arange(5, dtype=object)
        arg2 = arg1 + 1
        assert_equal(np.maximum(arg1, arg2), arg2)

    def test_strided_array(self):
        arr1 = np.array([-4.0, 1.0, 10.0, 0.0, np.nan, -np.nan, np.inf, -np.inf])
        arr2 = np.array([-2.0, -1.0, np.nan, 1.0, 0.0, np.nan, 1.0, -3.0])
        maxtrue = np.array([-2.0, 1.0, np.nan, 1.0, np.nan, np.nan, np.inf, -3.0])
        out = np.ones(8)
        out_maxtrue = np.array([-2.0, 1.0, 1.0, 10.0, 1.0, 1.0, np.nan, 1.0])
        assert_equal(np.maximum(arr1, arr2), maxtrue)
        assert_equal(np.maximum(arr1[::2], arr2[::2]), maxtrue[::2])
        assert_equal(np.maximum(arr1[:4], arr2[::2]), np.array([-2.0, np.nan, 10.0, 1.0]))
        assert_equal(np.maximum(arr1[::3], arr2[:3]), np.array([-2.0, 0.0, np.nan]))
        assert_equal(np.maximum(arr1[:6:2], arr2[::3], out=out[::3]), np.array([-2.0, 10.0, np.nan]))
        assert_equal(out, out_maxtrue)

    def test_precision(self):
        dtypes = [np.float16, np.float32, np.float64, np.longdouble]
        for dt in dtypes:
            dtmin = np.finfo(dt).min
            dtmax = np.finfo(dt).max
            d1 = dt(0.1)
            d1_next = np.nextafter(d1, np.inf)
            test_cases = [(dtmin, -np.inf, dtmin), (dtmax, -np.inf, dtmax), (d1, d1_next, d1_next), (dtmax, np.nan, np.nan)]
            for v1, v2, expected in test_cases:
                assert_equal(np.maximum([v1], [v2]), [expected])
                assert_equal(np.maximum.reduce([v1, v2]), expected)