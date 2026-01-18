import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
class TestIsclose:
    rtol = 1e-05
    atol = 1e-08

    def _setup(self):
        atol = self.atol
        rtol = self.rtol
        arr = np.array([100, 1000])
        aran = np.arange(125).reshape((5, 5, 5))
        self.all_close_tests = [([1, 0], [1, 0]), ([atol], [0]), ([1], [1 + rtol + atol]), (arr, arr + arr * rtol), (arr, arr + arr * rtol + atol), (aran, aran + aran * rtol), (np.inf, np.inf), (np.inf, [np.inf]), ([np.inf, -np.inf], [np.inf, -np.inf])]
        self.none_close_tests = [([np.inf, 0], [1, np.inf]), ([np.inf, -np.inf], [1, 0]), ([np.inf, np.inf], [1, -np.inf]), ([np.inf, np.inf], [1, 0]), ([np.nan, 0], [np.nan, -np.inf]), ([atol * 2], [0]), ([1], [1 + rtol + atol * 2]), (aran, aran + rtol * 1.1 * aran + atol * 1.1), (np.array([np.inf, 1]), np.array([0, np.inf]))]
        self.some_close_tests = [([np.inf, 0], [np.inf, atol * 2]), ([atol, 1, 1000000.0 * (1 + 2 * rtol) + atol], [0, np.nan, 1000000.0]), (np.arange(3), [0, 1, 2.1]), (np.nan, [np.nan, np.nan, np.nan]), ([0], [atol, np.inf, -np.inf, np.nan]), (0, [atol, np.inf, -np.inf, np.nan])]
        self.some_close_results = [[True, False], [True, False, False], [True, True, False], [False, False, False], [True, False, False, False], [True, False, False, False]]

    def test_ip_isclose(self):
        self._setup()
        tests = self.some_close_tests
        results = self.some_close_results
        for (x, y), result in zip(tests, results):
            assert_array_equal(np.isclose(x, y), result)

    def tst_all_isclose(self, x, y):
        assert_(np.all(np.isclose(x, y)), '%s and %s not close' % (x, y))

    def tst_none_isclose(self, x, y):
        msg = "%s and %s shouldn't be close"
        assert_(not np.any(np.isclose(x, y)), msg % (x, y))

    def tst_isclose_allclose(self, x, y):
        msg = "isclose.all() and allclose aren't same for %s and %s"
        msg2 = "isclose and allclose aren't same for %s and %s"
        if np.isscalar(x) and np.isscalar(y):
            assert_(np.isclose(x, y) == np.allclose(x, y), msg=msg2 % (x, y))
        else:
            assert_array_equal(np.isclose(x, y).all(), np.allclose(x, y), msg % (x, y))

    def test_ip_all_isclose(self):
        self._setup()
        for x, y in self.all_close_tests:
            self.tst_all_isclose(x, y)

    def test_ip_none_isclose(self):
        self._setup()
        for x, y in self.none_close_tests:
            self.tst_none_isclose(x, y)

    def test_ip_isclose_allclose(self):
        self._setup()
        tests = self.all_close_tests + self.none_close_tests + self.some_close_tests
        for x, y in tests:
            self.tst_isclose_allclose(x, y)

    def test_equal_nan(self):
        assert_array_equal(np.isclose(np.nan, np.nan, equal_nan=True), [True])
        arr = np.array([1.0, np.nan])
        assert_array_equal(np.isclose(arr, arr, equal_nan=True), [True, True])

    def test_masked_arrays(self):
        x = np.ma.masked_where([True, True, False], np.arange(3))
        assert_(type(x) is type(np.isclose(2, x)))
        assert_(type(x) is type(np.isclose(x, 2)))
        x = np.ma.masked_where([True, True, False], [np.nan, np.inf, np.nan])
        assert_(type(x) is type(np.isclose(np.inf, x)))
        assert_(type(x) is type(np.isclose(x, np.inf)))
        x = np.ma.masked_where([True, True, False], [np.nan, np.nan, np.nan])
        y = np.isclose(np.nan, x, equal_nan=True)
        assert_(type(x) is type(y))
        assert_array_equal([True, True, False], y.mask)
        y = np.isclose(x, np.nan, equal_nan=True)
        assert_(type(x) is type(y))
        assert_array_equal([True, True, False], y.mask)
        x = np.ma.masked_where([True, True, False], [np.nan, np.nan, np.nan])
        y = np.isclose(x, x, equal_nan=True)
        assert_(type(x) is type(y))
        assert_array_equal([True, True, False], y.mask)

    def test_scalar_return(self):
        assert_(np.isscalar(np.isclose(1, 1)))

    def test_no_parameter_modification(self):
        x = np.array([np.inf, 1])
        y = np.array([0, np.inf])
        np.isclose(x, y)
        assert_array_equal(x, np.array([np.inf, 1]))
        assert_array_equal(y, np.array([0, np.inf]))

    def test_non_finite_scalar(self):
        assert_(np.isclose(np.inf, -np.inf) is np.False_)
        assert_(np.isclose(0, np.inf) is np.False_)
        assert_(type(np.isclose(0, np.inf)) is np.bool_)

    def test_timedelta(self):
        a = np.array([[1, 2, 3, 'NaT']], dtype='m8[ns]')
        assert np.isclose(a, a, atol=0, equal_nan=True).all()
        assert np.isclose(a, a, atol=np.timedelta64(1, 'ns'), equal_nan=True).all()
        assert np.allclose(a, a, atol=0, equal_nan=True)
        assert np.allclose(a, a, atol=np.timedelta64(1, 'ns'), equal_nan=True)