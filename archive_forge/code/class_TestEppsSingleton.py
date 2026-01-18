from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc
class TestEppsSingleton:

    def test_statistic_1(self):
        x = np.array([-0.35, 2.55, 1.73, 0.73, 0.35, 2.69, 0.46, -0.94, -0.37, 12.07])
        y = np.array([-1.15, -0.15, 2.48, 3.25, 3.71, 4.29, 5.0, 7.74, 8.38, 8.6])
        w, p = epps_singleton_2samp(x, y)
        assert_almost_equal(w, 15.14, decimal=1)
        assert_almost_equal(p, 0.00442, decimal=3)

    def test_statistic_2(self):
        x = np.array((0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5, 5, 5, 6, 10, 10, 10, 10))
        y = np.array((10, 4, 0, 5, 10, 10, 0, 5, 6, 7, 10, 3, 1, 7, 0, 8, 1, 5, 8, 10))
        w, p = epps_singleton_2samp(x, y)
        assert_allclose(w, 8.9, atol=0.001)
        assert_almost_equal(p, 0.06364, decimal=3)

    def test_epps_singleton_array_like(self):
        np.random.seed(1234)
        x, y = (np.arange(30), np.arange(28))
        w1, p1 = epps_singleton_2samp(list(x), list(y))
        w2, p2 = epps_singleton_2samp(tuple(x), tuple(y))
        w3, p3 = epps_singleton_2samp(x, y)
        assert_(w1 == w2 == w3)
        assert_(p1 == p2 == p3)

    def test_epps_singleton_size(self):
        x, y = ((1, 2, 3, 4), np.arange(10))
        assert_raises(ValueError, epps_singleton_2samp, x, y)

    def test_epps_singleton_nonfinite(self):
        x, y = ((1, 2, 3, 4, 5, np.inf), np.arange(10))
        assert_raises(ValueError, epps_singleton_2samp, x, y)

    def test_names(self):
        x, y = (np.arange(20), np.arange(30))
        res = epps_singleton_2samp(x, y)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)