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
class TestBinomTest:
    """Tests for stats.binomtest."""

    def test_two_sided_pvalues1(self):
        rtol = 1e-10
        res = stats.binomtest(10079999, 21000000, 0.48)
        assert_allclose(res.pvalue, 1.0, rtol=rtol)
        res = stats.binomtest(10079990, 21000000, 0.48)
        assert_allclose(res.pvalue, 0.9966892187965, rtol=rtol)
        res = stats.binomtest(10080009, 21000000, 0.48)
        assert_allclose(res.pvalue, 0.9970377203856, rtol=rtol)
        res = stats.binomtest(10080017, 21000000, 0.48)
        assert_allclose(res.pvalue, 0.9940754817328, rtol=1e-09)

    def test_two_sided_pvalues2(self):
        rtol = 1e-10
        res = stats.binomtest(9, n=21, p=0.48)
        assert_allclose(res.pvalue, 0.6689672431939, rtol=rtol)
        res = stats.binomtest(4, 21, 0.48)
        assert_allclose(res.pvalue, 0.008139563452106, rtol=rtol)
        res = stats.binomtest(11, 21, 0.48)
        assert_allclose(res.pvalue, 0.8278629664608, rtol=rtol)
        res = stats.binomtest(7, 21, 0.48)
        assert_allclose(res.pvalue, 0.1966772901718, rtol=rtol)
        res = stats.binomtest(3, 10, 0.5)
        assert_allclose(res.pvalue, 0.34375, rtol=rtol)
        res = stats.binomtest(2, 2, 0.4)
        assert_allclose(res.pvalue, 0.16, rtol=rtol)
        res = stats.binomtest(2, 4, 0.3)
        assert_allclose(res.pvalue, 0.5884, rtol=rtol)

    def test_edge_cases(self):
        rtol = 1e-10
        res = stats.binomtest(484, 967, 0.5)
        assert_allclose(res.pvalue, 1, rtol=rtol)
        res = stats.binomtest(3, 47, 3 / 47)
        assert_allclose(res.pvalue, 1, rtol=rtol)
        res = stats.binomtest(13, 46, 13 / 46)
        assert_allclose(res.pvalue, 1, rtol=rtol)
        res = stats.binomtest(15, 44, 15 / 44)
        assert_allclose(res.pvalue, 1, rtol=rtol)
        res = stats.binomtest(7, 13, 0.5)
        assert_allclose(res.pvalue, 1, rtol=rtol)
        res = stats.binomtest(6, 11, 0.5)
        assert_allclose(res.pvalue, 1, rtol=rtol)

    def test_binary_srch_for_binom_tst(self):
        n = 10
        p = 0.5
        k = 3
        i = np.arange(np.ceil(p * n), n + 1)
        d = stats.binom.pmf(k, n, p)
        y1 = np.sum(stats.binom.pmf(i, n, p) <= d, axis=0)
        ix = _binary_search_for_binom_tst(lambda x1: -stats.binom.pmf(x1, n, p), -d, np.ceil(p * n), n)
        y2 = n - ix + int(d == stats.binom.pmf(ix, n, p))
        assert_allclose(y1, y2, rtol=1e-09)
        k = 7
        i = np.arange(np.floor(p * n) + 1)
        d = stats.binom.pmf(k, n, p)
        y1 = np.sum(stats.binom.pmf(i, n, p) <= d, axis=0)
        ix = _binary_search_for_binom_tst(lambda x1: stats.binom.pmf(x1, n, p), d, 0, np.floor(p * n))
        y2 = ix + 1
        assert_allclose(y1, y2, rtol=1e-09)

    @pytest.mark.parametrize('alternative, pval, ci_low, ci_high', [('less', 0.148831050443, 0.0, 0.2772002496709138), ('greater', 0.9004695898947, 0.1366613252458672, 1.0), ('two-sided', 0.2983720970096, 0.1266555521019559, 0.2918426890886281)])
    def test_confidence_intervals1(self, alternative, pval, ci_low, ci_high):
        res = stats.binomtest(20, n=100, p=0.25, alternative=alternative)
        assert_allclose(res.pvalue, pval, rtol=1e-12)
        assert_equal(res.statistic, 0.2)
        ci = res.proportion_ci(confidence_level=0.95)
        assert_allclose((ci.low, ci.high), (ci_low, ci_high), rtol=1e-12)

    @pytest.mark.parametrize('alternative, pval, ci_low, ci_high', [('less', 0.005656361, 0.0, 0.1872093), ('greater', 0.9987146, 0.008860761, 1.0), ('two-sided', 0.01191714, 0.006872485, 0.202706269)])
    def test_confidence_intervals2(self, alternative, pval, ci_low, ci_high):
        res = stats.binomtest(3, n=50, p=0.2, alternative=alternative)
        assert_allclose(res.pvalue, pval, rtol=1e-06)
        assert_equal(res.statistic, 0.06)
        ci = res.proportion_ci(confidence_level=0.99)
        assert_allclose((ci.low, ci.high), (ci_low, ci_high), rtol=1e-06)

    @pytest.mark.parametrize('alternative, pval, ci_high', [('less', 0.05631351, 0.2588656), ('greater', 1.0, 1.0), ('two-sided', 0.07604122, 0.3084971)])
    def test_confidence_interval_exact_k0(self, alternative, pval, ci_high):
        res = stats.binomtest(0, 10, p=0.25, alternative=alternative)
        assert_allclose(res.pvalue, pval, rtol=1e-06)
        ci = res.proportion_ci(confidence_level=0.95)
        assert_equal(ci.low, 0.0)
        assert_allclose(ci.high, ci_high, rtol=1e-06)

    @pytest.mark.parametrize('alternative, pval, ci_low', [('less', 1.0, 0.0), ('greater', 9.536743e-07, 0.7411344), ('two-sided', 9.536743e-07, 0.6915029)])
    def test_confidence_interval_exact_k_is_n(self, alternative, pval, ci_low):
        res = stats.binomtest(10, 10, p=0.25, alternative=alternative)
        assert_allclose(res.pvalue, pval, rtol=1e-06)
        ci = res.proportion_ci(confidence_level=0.95)
        assert_equal(ci.high, 1.0)
        assert_allclose(ci.low, ci_low, rtol=1e-06)

    @pytest.mark.parametrize('k, alternative, corr, conf, ci_low, ci_high', [[3, 'two-sided', True, 0.95, 0.08094782, 0.64632928], [3, 'two-sided', True, 0.99, 0.0586329, 0.7169416], [3, 'two-sided', False, 0.95, 0.1077913, 0.6032219], [3, 'two-sided', False, 0.99, 0.07956632, 0.6799753], [3, 'less', True, 0.95, 0.0, 0.6043476], [3, 'less', True, 0.99, 0.0, 0.6901811], [3, 'less', False, 0.95, 0.0, 0.5583002], [3, 'less', False, 0.99, 0.0, 0.6507187], [3, 'greater', True, 0.95, 0.09644904, 1.0], [3, 'greater', True, 0.99, 0.06659141, 1.0], [3, 'greater', False, 0.95, 0.1268766, 1.0], [3, 'greater', False, 0.99, 0.08974147, 1.0], [0, 'two-sided', True, 0.95, 0.0, 0.3445372], [0, 'two-sided', False, 0.95, 0.0, 0.2775328], [0, 'less', True, 0.95, 0.0, 0.2847374], [0, 'less', False, 0.95, 0.0, 0.212942], [0, 'greater', True, 0.95, 0.0, 1.0], [0, 'greater', False, 0.95, 0.0, 1.0], [10, 'two-sided', True, 0.95, 0.6554628, 1.0], [10, 'two-sided', False, 0.95, 0.7224672, 1.0], [10, 'less', True, 0.95, 0.0, 1.0], [10, 'less', False, 0.95, 0.0, 1.0], [10, 'greater', True, 0.95, 0.7152626, 1.0], [10, 'greater', False, 0.95, 0.787058, 1.0]])
    def test_ci_wilson_method(self, k, alternative, corr, conf, ci_low, ci_high):
        res = stats.binomtest(k, n=10, p=0.1, alternative=alternative)
        if corr:
            method = 'wilsoncc'
        else:
            method = 'wilson'
        ci = res.proportion_ci(confidence_level=conf, method=method)
        assert_allclose((ci.low, ci.high), (ci_low, ci_high), rtol=1e-06)

    def test_estimate_equals_hypothesized_prop(self):
        res = stats.binomtest(4, 16, 0.25)
        assert_equal(res.statistic, 0.25)
        assert_equal(res.pvalue, 1.0)

    @pytest.mark.parametrize('k, n', [(0, 0), (-1, 2)])
    def test_invalid_k_n(self, k, n):
        with pytest.raises(ValueError, match='must be an integer not less than'):
            stats.binomtest(k, n)

    def test_invalid_k_too_big(self):
        with pytest.raises(ValueError, match='k \\(11\\) must not be greater than n \\(10\\).'):
            stats.binomtest(11, 10, 0.25)

    def test_invalid_k_wrong_type(self):
        with pytest.raises(TypeError, match='k must be an integer.'):
            stats.binomtest([10, 11], 21, 0.25)

    def test_invalid_p_range(self):
        message = 'p \\(-0.5\\) must be in range...'
        with pytest.raises(ValueError, match=message):
            stats.binomtest(50, 150, p=-0.5)
        message = 'p \\(1.5\\) must be in range...'
        with pytest.raises(ValueError, match=message):
            stats.binomtest(50, 150, p=1.5)

    def test_invalid_confidence_level(self):
        res = stats.binomtest(3, n=10, p=0.1)
        message = 'confidence_level \\(-1\\) must be in the interval'
        with pytest.raises(ValueError, match=message):
            res.proportion_ci(confidence_level=-1)

    def test_invalid_ci_method(self):
        res = stats.binomtest(3, n=10, p=0.1)
        with pytest.raises(ValueError, match="method \\('plate of shrimp'\\) must be"):
            res.proportion_ci(method='plate of shrimp')

    def test_invalid_alternative(self):
        with pytest.raises(ValueError, match="alternative \\('ekki'\\) not..."):
            stats.binomtest(3, n=10, p=0.1, alternative='ekki')

    def test_alias(self):
        res = stats.binomtest(3, n=10, p=0.1)
        assert_equal(res.proportion_estimate, res.statistic)

    @pytest.mark.skipif(sys.maxsize <= 2 ** 32, reason='32-bit does not overflow')
    def test_boost_overflow_raises(self):
        with pytest.raises(OverflowError, match='Error in function...'):
            stats.binomtest(5, 6, p=sys.float_info.min)