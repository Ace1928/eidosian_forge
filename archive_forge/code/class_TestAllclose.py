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
class TestAllclose:
    rtol = 1e-05
    atol = 1e-08

    def setup_method(self):
        self.olderr = np.seterr(invalid='ignore')

    def teardown_method(self):
        np.seterr(**self.olderr)

    def tst_allclose(self, x, y):
        assert_(np.allclose(x, y), '%s and %s not close' % (x, y))

    def tst_not_allclose(self, x, y):
        assert_(not np.allclose(x, y), "%s and %s shouldn't be close" % (x, y))

    def test_ip_allclose(self):
        arr = np.array([100, 1000])
        aran = np.arange(125).reshape((5, 5, 5))
        atol = self.atol
        rtol = self.rtol
        data = [([1, 0], [1, 0]), ([atol], [0]), ([1], [1 + rtol + atol]), (arr, arr + arr * rtol), (arr, arr + arr * rtol + atol * 2), (aran, aran + aran * rtol), (np.inf, np.inf), (np.inf, [np.inf])]
        for x, y in data:
            self.tst_allclose(x, y)

    def test_ip_not_allclose(self):
        aran = np.arange(125).reshape((5, 5, 5))
        atol = self.atol
        rtol = self.rtol
        data = [([np.inf, 0], [1, np.inf]), ([np.inf, 0], [1, 0]), ([np.inf, np.inf], [1, np.inf]), ([np.inf, np.inf], [1, 0]), ([-np.inf, 0], [np.inf, 0]), ([np.nan, 0], [np.nan, 0]), ([atol * 2], [0]), ([1], [1 + rtol + atol * 2]), (aran, aran + aran * atol + atol * 2), (np.array([np.inf, 1]), np.array([0, np.inf]))]
        for x, y in data:
            self.tst_not_allclose(x, y)

    def test_no_parameter_modification(self):
        x = np.array([np.inf, 1])
        y = np.array([0, np.inf])
        np.allclose(x, y)
        assert_array_equal(x, np.array([np.inf, 1]))
        assert_array_equal(y, np.array([0, np.inf]))

    def test_min_int(self):
        min_int = np.iinfo(np.int_).min
        a = np.array([min_int], dtype=np.int_)
        assert_(np.allclose(a, a))

    def test_equalnan(self):
        x = np.array([1.0, np.nan])
        assert_(np.allclose(x, x, equal_nan=True))

    def test_return_class_is_ndarray(self):

        class Foo(np.ndarray):

            def __new__(cls, *args, **kwargs):
                return np.array(*args, **kwargs).view(cls)
        a = Foo([1])
        assert_(type(np.allclose(a, a)) is bool)