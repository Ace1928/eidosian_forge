import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
class TestMedianTest:

    def test_bad_n_samples(self):
        assert_raises(ValueError, stats.median_test, [1, 2, 3])

    def test_empty_sample(self):
        assert_raises(ValueError, stats.median_test, [], [1, 2, 3])

    def test_empty_when_ties_ignored(self):
        assert_raises(ValueError, stats.median_test, [1, 1, 1, 1], [2, 0, 1], [2, 0], ties='ignore')

    def test_empty_contingency_row(self):
        assert_raises(ValueError, stats.median_test, [1, 1, 1], [1, 1, 1])
        assert_raises(ValueError, stats.median_test, [1, 1, 1], [1, 1, 1], ties='above')

    def test_bad_ties(self):
        assert_raises(ValueError, stats.median_test, [1, 2, 3], [4, 5], ties='foo')

    def test_bad_nan_policy(self):
        assert_raises(ValueError, stats.median_test, [1, 2, 3], [4, 5], nan_policy='foobar')

    def test_bad_keyword(self):
        assert_raises(TypeError, stats.median_test, [1, 2, 3], [4, 5], foo='foo')

    def test_simple(self):
        x = [1, 2, 3]
        y = [1, 2, 3]
        stat, p, med, tbl = stats.median_test(x, y)
        assert_equal(med, 2.0)
        assert_array_equal(tbl, [[1, 1], [2, 2]])
        assert_equal(stat, 0)
        assert_equal(p, 1)

    def test_ties_options(self):
        x = [1, 2, 3, 4]
        y = [5, 6]
        z = [7, 8, 9]
        stat, p, m, tbl = stats.median_test(x, y, z)
        assert_equal(m, 5)
        assert_equal(tbl, [[0, 1, 3], [4, 1, 0]])
        stat, p, m, tbl = stats.median_test(x, y, z, ties='ignore')
        assert_equal(m, 5)
        assert_equal(tbl, [[0, 1, 3], [4, 0, 0]])
        stat, p, m, tbl = stats.median_test(x, y, z, ties='above')
        assert_equal(m, 5)
        assert_equal(tbl, [[0, 2, 3], [4, 0, 0]])

    def test_nan_policy_options(self):
        x = [1, 2, np.nan]
        y = [4, 5, 6]
        mt1 = stats.median_test(x, y, nan_policy='propagate')
        s, p, m, t = stats.median_test(x, y, nan_policy='omit')
        assert_equal(mt1, (np.nan, np.nan, np.nan, None))
        assert_allclose(s, 0.31250000000000006)
        assert_allclose(p, 0.5761501220305787)
        assert_equal(m, 4.0)
        assert_equal(t, np.array([[0, 2], [2, 1]]))
        assert_raises(ValueError, stats.median_test, x, y, nan_policy='raise')

    def test_basic(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8]
        stat, p, m, tbl = stats.median_test(x, y)
        assert_equal(m, 4)
        assert_equal(tbl, [[1, 2], [4, 2]])
        exp_stat, exp_p, dof, e = stats.chi2_contingency(tbl)
        assert_allclose(stat, exp_stat)
        assert_allclose(p, exp_p)
        stat, p, m, tbl = stats.median_test(x, y, lambda_=0)
        assert_equal(m, 4)
        assert_equal(tbl, [[1, 2], [4, 2]])
        exp_stat, exp_p, dof, e = stats.chi2_contingency(tbl, lambda_=0)
        assert_allclose(stat, exp_stat)
        assert_allclose(p, exp_p)
        stat, p, m, tbl = stats.median_test(x, y, correction=False)
        assert_equal(m, 4)
        assert_equal(tbl, [[1, 2], [4, 2]])
        exp_stat, exp_p, dof, e = stats.chi2_contingency(tbl, correction=False)
        assert_allclose(stat, exp_stat)
        assert_allclose(p, exp_p)

    @pytest.mark.parametrize('correction', [False, True])
    def test_result(self, correction):
        x = [1, 2, 3]
        y = [1, 2, 3]
        res = stats.median_test(x, y, correction=correction)
        assert_equal((res.statistic, res.pvalue, res.median, res.table), res)