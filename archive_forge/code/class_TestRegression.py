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
class TestRegression:

    def test_linregressBIGX(self):
        result = stats.linregress(X, BIG)
        assert_almost_equal(result.intercept, 99999990)
        assert_almost_equal(result.rvalue, 1.0)
        assert_almost_equal(result.stderr, 0.0)
        assert_almost_equal(result.intercept_stderr, 0.0)

    def test_regressXX(self):
        result = stats.linregress(X, X)
        assert_almost_equal(result.intercept, 0.0)
        assert_almost_equal(result.rvalue, 1.0)
        assert_almost_equal(result.stderr, 0.0)
        assert_almost_equal(result.intercept_stderr, 0.0)

    def test_regressZEROX(self):
        result = stats.linregress(X, ZERO)
        assert_almost_equal(result.intercept, 0.0)
        assert_almost_equal(result.rvalue, 0.0)

    def test_regress_simple(self):
        x = np.linspace(0, 100, 100)
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))
        result = stats.linregress(x, y)
        lr = stats._stats_mstats_common.LinregressResult
        assert_(isinstance(result, lr))
        assert_almost_equal(result.stderr, 0.0023957814497838803)

    def test_regress_alternative(self):
        x = np.linspace(0, 100, 100)
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))
        with pytest.raises(ValueError, match="alternative must be 'less'..."):
            stats.linregress(x, y, alternative='ekki-ekki')
        res1 = stats.linregress(x, y, alternative='two-sided')
        res2 = stats.linregress(x, y, alternative='less')
        assert_allclose(res2.pvalue, 1 - res1.pvalue / 2)
        res3 = stats.linregress(x, y, alternative='greater')
        assert_allclose(res3.pvalue, res1.pvalue / 2)
        assert res1.rvalue == res2.rvalue == res3.rvalue

    def test_regress_against_R(self):
        x = [151, 174, 138, 186, 128, 136, 179, 163, 152, 131]
        y = [63, 81, 56, 91, 47, 57, 76, 72, 62, 48]
        res = stats.linregress(x, y, alternative='two-sided')
        assert_allclose(res.slope, 0.6746104491292)
        assert_allclose(res.intercept, -38.455087076077)
        assert_allclose(res.rvalue, np.sqrt(0.95478224775))
        assert_allclose(res.pvalue, 1.16440531074e-06)
        assert_allclose(res.stderr, 0.0519051424731)
        assert_allclose(res.intercept_stderr, 8.0490133029927)

    def test_regress_simple_onearg_rows(self):
        x = np.linspace(0, 100, 100)
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))
        rows = np.vstack((x, y))
        result = stats.linregress(rows)
        assert_almost_equal(result.stderr, 0.0023957814497838803)
        assert_almost_equal(result.intercept_stderr, 0.13866936078570702)

    def test_regress_simple_onearg_cols(self):
        x = np.linspace(0, 100, 100)
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))
        columns = np.hstack((np.expand_dims(x, 1), np.expand_dims(y, 1)))
        result = stats.linregress(columns)
        assert_almost_equal(result.stderr, 0.0023957814497838803)
        assert_almost_equal(result.intercept_stderr, 0.13866936078570702)

    def test_regress_shape_error(self):
        assert_raises(ValueError, stats.linregress, np.ones((3, 3)))

    def test_linregress(self):
        x = np.arange(11)
        y = np.arange(5, 16)
        y[[1, -2]] -= 1
        y[[0, -1]] += 1
        result = stats.linregress(x, y)

        def assert_ae(x, y):
            return assert_almost_equal(x, y, decimal=14)
        assert_ae(result.slope, 1.0)
        assert_ae(result.intercept, 5.0)
        assert_ae(result.rvalue, 0.9822994862575)
        assert_ae(result.pvalue, 7.45259691e-08)
        assert_ae(result.stderr, 0.06356417261637273)
        assert_ae(result.intercept_stderr, 0.37605071654517686)

    def test_regress_simple_negative_cor(self):
        a, n = (1e-71, 100000)
        x = np.linspace(a, 2 * a, n)
        y = np.linspace(2 * a, a, n)
        result = stats.linregress(x, y)
        assert_(result.rvalue >= -1)
        assert_almost_equal(result.rvalue, -1)
        assert_(not np.isnan(result.stderr))
        assert_(not np.isnan(result.intercept_stderr))

    def test_linregress_result_attributes(self):
        x = np.linspace(0, 100, 100)
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))
        result = stats.linregress(x, y)
        lr = stats._stats_mstats_common.LinregressResult
        assert_(isinstance(result, lr))
        attributes = ('slope', 'intercept', 'rvalue', 'pvalue', 'stderr')
        check_named_results(result, attributes)
        assert 'intercept_stderr' in dir(result)

    def test_regress_two_inputs(self):
        x = np.arange(2)
        y = np.arange(3, 5)
        result = stats.linregress(x, y)
        assert_almost_equal(result.pvalue, 0.0)
        assert_almost_equal(result.stderr, 0.0)
        assert_almost_equal(result.intercept_stderr, 0.0)

    def test_regress_two_inputs_horizontal_line(self):
        x = np.arange(2)
        y = np.ones(2)
        result = stats.linregress(x, y)
        assert_almost_equal(result.pvalue, 1.0)
        assert_almost_equal(result.stderr, 0.0)
        assert_almost_equal(result.intercept_stderr, 0.0)

    def test_nist_norris(self):
        x = [0.2, 337.4, 118.2, 884.6, 10.1, 226.5, 666.3, 996.3, 448.6, 777.0, 558.2, 0.4, 0.6, 775.5, 666.9, 338.0, 447.5, 11.6, 556.0, 228.1, 995.8, 887.6, 120.2, 0.3, 0.3, 556.8, 339.1, 887.2, 999.0, 779.0, 11.1, 118.3, 229.2, 669.1, 448.9, 0.5]
        y = [0.1, 338.8, 118.1, 888.0, 9.2, 228.1, 668.5, 998.5, 449.1, 778.9, 559.2, 0.3, 0.1, 778.1, 668.8, 339.3, 448.9, 10.8, 557.7, 228.3, 998.0, 888.8, 119.6, 0.3, 0.6, 557.6, 339.3, 888.0, 998.5, 778.9, 10.2, 117.6, 228.9, 668.4, 449.2, 0.2]
        result = stats.linregress(x, y)
        assert_almost_equal(result.slope, 1.00211681802045)
        assert_almost_equal(result.intercept, -0.262323073774029)
        assert_almost_equal(result.rvalue ** 2, 0.999993745883712)
        assert_almost_equal(result.pvalue, 0.0)
        assert_almost_equal(result.stderr, 0.0004297968482)
        assert_almost_equal(result.intercept_stderr, 0.23281823430153)

    def test_compare_to_polyfit(self):
        x = np.linspace(0, 100, 100)
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))
        result = stats.linregress(x, y)
        poly = np.polyfit(x, y, 1)
        assert_almost_equal(result.slope, poly[0])
        assert_almost_equal(result.intercept, poly[1])

    def test_empty_input(self):
        assert_raises(ValueError, stats.linregress, [], [])

    def test_nan_input(self):
        x = np.arange(10.0)
        x[9] = np.nan
        with np.errstate(invalid='ignore'):
            result = stats.linregress(x, x)
        lr = stats._stats_mstats_common.LinregressResult
        assert_(isinstance(result, lr))
        assert_array_equal(result, (np.nan,) * 5)
        assert_equal(result.intercept_stderr, np.nan)

    def test_identical_x(self):
        x = np.zeros(10)
        y = np.random.random(10)
        msg = 'Cannot calculate a linear regression'
        with assert_raises(ValueError, match=msg):
            stats.linregress(x, y)