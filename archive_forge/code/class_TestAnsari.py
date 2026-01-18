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
class TestAnsari:

    def test_small(self):
        x = [1, 2, 3, 3, 4]
        y = [3, 2, 6, 1, 6, 1, 4, 1]
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'Ties preclude use of exact statistic.')
            W, pval = stats.ansari(x, y)
        assert_almost_equal(W, 23.5, 11)
        assert_almost_equal(pval, 0.13499256881897437, 11)

    def test_approx(self):
        ramsay = np.array((111, 107, 100, 99, 102, 106, 109, 108, 104, 99, 101, 96, 97, 102, 107, 113, 116, 113, 110, 98))
        parekh = np.array((107, 108, 106, 98, 105, 103, 110, 105, 104, 100, 96, 108, 103, 104, 114, 114, 113, 108, 106, 99))
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'Ties preclude use of exact statistic.')
            W, pval = stats.ansari(ramsay, parekh)
        assert_almost_equal(W, 185.5, 11)
        assert_almost_equal(pval, 0.18145819972867083, 11)

    def test_exact(self):
        W, pval = stats.ansari([1, 2, 3, 4], [15, 5, 20, 8, 10, 12])
        assert_almost_equal(W, 10.0, 11)
        assert_almost_equal(pval, 0.5333333333333333, 7)

    def test_bad_arg(self):
        assert_raises(ValueError, stats.ansari, [], [1])
        assert_raises(ValueError, stats.ansari, [1], [])

    def test_result_attributes(self):
        x = [1, 2, 3, 3, 4]
        y = [3, 2, 6, 1, 6, 1, 4, 1]
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'Ties preclude use of exact statistic.')
            res = stats.ansari(x, y)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

    def test_bad_alternative(self):
        x1 = [1, 2, 3, 4]
        x2 = [5, 6, 7, 8]
        match = "'alternative' must be 'two-sided'"
        with assert_raises(ValueError, match=match):
            stats.ansari(x1, x2, alternative='foo')

    def test_alternative_exact(self):
        x1 = [-5, 1, 5, 10, 15, 20, 25]
        x2 = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
        statistic, pval = stats.ansari(x1, x2)
        pval_l = stats.ansari(x1, x2, alternative='less').pvalue
        pval_g = stats.ansari(x1, x2, alternative='greater').pvalue
        assert pval_l > 0.95
        assert pval_g < 0.05
        prob = _abw_state.pmf(statistic, len(x1), len(x2))
        assert_allclose(pval_g + pval_l, 1 + prob, atol=1e-12)
        assert_allclose(pval_g, pval / 2, atol=1e-12)
        assert_allclose(pval_l, 1 + prob - pval / 2, atol=1e-12)
        pval_l_reverse = stats.ansari(x2, x1, alternative='less').pvalue
        pval_g_reverse = stats.ansari(x2, x1, alternative='greater').pvalue
        assert pval_l_reverse < 0.05
        assert pval_g_reverse > 0.95

    @pytest.mark.parametrize('x, y, alternative, expected', [([1, 2, 3, 4], [5, 6, 7, 8], 'less', 0.6285714285714), ([1, 2, 3, 4], [5, 6, 7, 8], 'greater', 0.6285714285714), ([1, 2, 3], [4, 5, 6, 7, 8], 'less', 0.8928571428571), ([1, 2, 3], [4, 5, 6, 7, 8], 'greater', 0.2857142857143), ([1, 2, 3, 4, 5], [6, 7, 8], 'less', 0.2857142857143), ([1, 2, 3, 4, 5], [6, 7, 8], 'greater', 0.8928571428571)])
    def test_alternative_exact_with_R(self, x, y, alternative, expected):
        pval = stats.ansari(x, y, alternative=alternative).pvalue
        assert_allclose(pval, expected, atol=1e-12)

    def test_alternative_approx(self):
        x1 = stats.norm.rvs(0, 5, size=100, random_state=123)
        x2 = stats.norm.rvs(0, 2, size=100, random_state=123)
        pval_l = stats.ansari(x1, x2, alternative='less').pvalue
        pval_g = stats.ansari(x1, x2, alternative='greater').pvalue
        assert_allclose(pval_l, 1.0, atol=1e-12)
        assert_allclose(pval_g, 0.0, atol=1e-12)
        x1 = stats.norm.rvs(0, 2, size=60, random_state=123)
        x2 = stats.norm.rvs(0, 1.5, size=60, random_state=123)
        pval = stats.ansari(x1, x2).pvalue
        pval_l = stats.ansari(x1, x2, alternative='less').pvalue
        pval_g = stats.ansari(x1, x2, alternative='greater').pvalue
        assert_allclose(pval_g, pval / 2, atol=1e-12)
        assert_allclose(pval_l, 1 - pval / 2, atol=1e-12)