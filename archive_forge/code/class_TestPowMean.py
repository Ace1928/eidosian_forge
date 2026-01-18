import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
class TestPowMean:

    def pmean_reference(a, p):
        return (np.sum(a ** p) / a.size) ** (1 / p)

    def wpmean_reference(a, p, weights):
        return (np.sum(weights * a ** p) / np.sum(weights)) ** (1 / p)

    def test_bad_exponent(self):
        with pytest.raises(ValueError, match='Power mean only defined for'):
            stats.pmean([1, 2, 3], [0])
        with pytest.raises(ValueError, match='Power mean only defined for'):
            stats.pmean([1, 2, 3], np.array([0]))

    def test_1d_list(self):
        a, p = ([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 3.5)
        desired = TestPowMean.pmean_reference(np.array(a), p)
        check_equal_pmean(a, p, desired)
        a, p = ([1, 2, 3, 4], 2)
        desired = np.sqrt((1 ** 2 + 2 ** 2 + 3 ** 2 + 4 ** 2) / 4)
        check_equal_pmean(a, p, desired)

    def test_1d_array(self):
        a, p = (np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), -2.5)
        desired = TestPowMean.pmean_reference(a, p)
        check_equal_pmean(a, p, desired)

    def test_1d_array_with_zero(self):
        a, p = (np.array([1, 0]), -1)
        desired = 0.0
        assert_equal(stats.pmean(a, p), desired)

    def test_1d_array_with_negative_value(self):
        a, p = (np.array([1, 0, -1]), 1.23)
        with pytest.raises(ValueError, match='Power mean only defined if all'):
            stats.pmean(a, p)

    @pytest.mark.parametrize(('a', 'p'), [([[10, 20], [50, 60], [90, 100]], -0.5), (np.array([[10, 20], [50, 60], [90, 100]]), 0.5)])
    def test_2d_axisnone(self, a, p):
        desired = TestPowMean.pmean_reference(np.array(a), p)
        check_equal_pmean(a, p, desired)

    @pytest.mark.parametrize(('a', 'p'), [([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], -0.5), ([[10, 0, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], 0.5)])
    def test_2d_list_axis0(self, a, p):
        desired = [TestPowMean.pmean_reference(np.array([a[i][j] for i in range(len(a))]), p) for j in range(len(a[0]))]
        check_equal_pmean(a, p, desired, axis=0)

    @pytest.mark.parametrize(('a', 'p'), [([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], -0.5), ([[10, 0, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], 0.5)])
    def test_2d_list_axis1(self, a, p):
        desired = [TestPowMean.pmean_reference(np.array(a_), p) for a_ in a]
        check_equal_pmean(a, p, desired, axis=1)

    def test_weights_1d_list(self):
        a, p = ([2, 10, 6], -1.23456789)
        weights = [10, 5, 3]
        desired = TestPowMean.wpmean_reference(np.array(a), p, weights)
        check_equal_pmean(a, p, desired, weights=weights, rtol=1e-05)

    def test_weights_masked_1d_array(self):
        a, p = (np.array([2, 10, 6, 42]), 1)
        weights = np.ma.array([10, 5, 3, 42], mask=[0, 0, 0, 1])
        desired = np.average(a, weights=weights)
        check_equal_pmean(a, p, desired, weights=weights, rtol=1e-05)

    @pytest.mark.parametrize(('axis', 'fun_name', 'p'), [(None, 'wpmean_reference', 9.87654321), (0, 'gmean', 0), (1, 'hmean', -1)])
    def test_weights_2d_array(self, axis, fun_name, p):
        if fun_name == 'wpmean_reference':

            def fun(a, axis, weights):
                return TestPowMean.wpmean_reference(a, p, weights)
        else:
            fun = getattr(stats, fun_name)
        a = np.array([[2, 5], [10, 5], [6, 5]])
        weights = np.array([[10, 1], [5, 1], [3, 1]])
        desired = fun(a, axis=axis, weights=weights)
        check_equal_pmean(a, p, desired, axis=axis, weights=weights, rtol=1e-05)