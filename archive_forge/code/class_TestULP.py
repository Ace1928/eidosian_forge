import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
class TestULP:

    def test_equal(self):
        x = np.random.randn(10)
        assert_array_max_ulp(x, x, maxulp=0)

    def test_single(self):
        x = np.ones(10).astype(np.float32)
        x += 0.01 * np.random.randn(10).astype(np.float32)
        eps = np.finfo(np.float32).eps
        assert_array_max_ulp(x, x + eps, maxulp=20)

    def test_double(self):
        x = np.ones(10).astype(np.float64)
        x += 0.01 * np.random.randn(10).astype(np.float64)
        eps = np.finfo(np.float64).eps
        assert_array_max_ulp(x, x + eps, maxulp=200)

    def test_inf(self):
        for dt in [np.float32, np.float64]:
            inf = np.array([np.inf]).astype(dt)
            big = np.array([np.finfo(dt).max])
            assert_array_max_ulp(inf, big, maxulp=200)

    def test_nan(self):
        for dt in [np.float32, np.float64]:
            if dt == np.float32:
                maxulp = 1000000.0
            else:
                maxulp = 1000000000000.0
            inf = np.array([np.inf]).astype(dt)
            nan = np.array([np.nan]).astype(dt)
            big = np.array([np.finfo(dt).max])
            tiny = np.array([np.finfo(dt).tiny])
            zero = np.array([np.PZERO]).astype(dt)
            nzero = np.array([np.NZERO]).astype(dt)
            assert_raises(AssertionError, lambda: assert_array_max_ulp(nan, inf, maxulp=maxulp))
            assert_raises(AssertionError, lambda: assert_array_max_ulp(nan, big, maxulp=maxulp))
            assert_raises(AssertionError, lambda: assert_array_max_ulp(nan, tiny, maxulp=maxulp))
            assert_raises(AssertionError, lambda: assert_array_max_ulp(nan, zero, maxulp=maxulp))
            assert_raises(AssertionError, lambda: assert_array_max_ulp(nan, nzero, maxulp=maxulp))