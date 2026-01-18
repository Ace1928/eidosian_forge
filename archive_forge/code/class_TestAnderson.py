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
class TestAnderson:

    def test_normal(self):
        rs = RandomState(1234567890)
        x1 = rs.standard_exponential(size=50)
        x2 = rs.standard_normal(size=50)
        A, crit, sig = stats.anderson(x1)
        assert_array_less(crit[:-1], A)
        A, crit, sig = stats.anderson(x2)
        assert_array_less(A, crit[-2:])
        v = np.ones(10)
        v[0] = 0
        A, crit, sig = stats.anderson(v)
        assert_allclose(A, 3.208057)

    def test_expon(self):
        rs = RandomState(1234567890)
        x1 = rs.standard_exponential(size=50)
        x2 = rs.standard_normal(size=50)
        A, crit, sig = stats.anderson(x1, 'expon')
        assert_array_less(A, crit[-2:])
        with np.errstate(all='ignore'):
            A, crit, sig = stats.anderson(x2, 'expon')
        assert_(A > crit[-1])

    def test_gumbel(self):
        v = np.ones(100)
        v[0] = 0.0
        a2, crit, sig = stats.anderson(v, 'gumbel')
        n = len(v)
        xbar, s = stats.gumbel_l.fit(v)
        logcdf = stats.gumbel_l.logcdf(v, xbar, s)
        logsf = stats.gumbel_l.logsf(v, xbar, s)
        i = np.arange(1, n + 1)
        expected_a2 = -n - np.mean((2 * i - 1) * (logcdf + logsf[::-1]))
        assert_allclose(a2, expected_a2)

    def test_bad_arg(self):
        assert_raises(ValueError, stats.anderson, [1], dist='plate_of_shrimp')

    def test_result_attributes(self):
        rs = RandomState(1234567890)
        x = rs.standard_exponential(size=50)
        res = stats.anderson(x)
        attributes = ('statistic', 'critical_values', 'significance_level')
        check_named_results(res, attributes)

    def test_gumbel_l(self):
        rs = RandomState(1234567890)
        x = rs.gumbel(size=100)
        A1, crit1, sig1 = stats.anderson(x, 'gumbel')
        A2, crit2, sig2 = stats.anderson(x, 'gumbel_l')
        assert_allclose(A2, A1)

    def test_gumbel_r(self):
        rs = RandomState(1234567890)
        x1 = rs.gumbel(size=100)
        x2 = np.ones(100)
        x2[0] = 0.996
        A1, crit1, sig1 = stats.anderson(x1, 'gumbel_r')
        A2, crit2, sig2 = stats.anderson(x2, 'gumbel_r')
        assert_array_less(A1, crit1[-2:])
        assert_(A2 > crit2[-1])

    def test_weibull_min_case_A(self):
        x = np.array([225, 171, 198, 189, 189, 135, 162, 135, 117, 162])
        res = stats.anderson(x, 'weibull_min')
        m, loc, scale = res.fit_result.params
        assert_allclose((m, loc, scale), (2.38, 99.02, 78.23), rtol=0.002)
        assert_allclose(res.statistic, 0.26, rtol=0.001)
        assert res.statistic < res.critical_values[0]
        c = 1 / m
        assert_allclose(c, 1 / 2.38, rtol=0.002)
        As40 = _Avals_weibull[-3]
        As45 = _Avals_weibull[-2]
        As_ref = As40 + (c - 0.4) / (0.45 - 0.4) * (As45 - As40)
        assert np.all(res.critical_values > As_ref)
        assert_allclose(res.critical_values, As_ref, atol=0.001)

    def test_weibull_min_case_B(self):
        x = np.array([74, 57, 48, 29, 502, 12, 70, 21, 29, 386, 59, 27, 153, 26, 326])
        message = 'Maximum likelihood estimation has converged to '
        with pytest.raises(ValueError, match=message):
            stats.anderson(x, 'weibull_min')

    def test_weibull_warning_error(self):
        x = -np.array([225, 75, 57, 168, 107, 12, 61, 43, 29])
        wmessage = 'Critical values of the test statistic are given for the...'
        emessage = 'An error occurred while fitting the Weibull distribution...'
        wcontext = pytest.warns(UserWarning, match=wmessage)
        econtext = pytest.raises(ValueError, match=emessage)
        with wcontext, econtext:
            stats.anderson(x, 'weibull_min')

    @pytest.mark.parametrize('distname', ['norm', 'expon', 'gumbel_l', 'extreme1', 'gumbel', 'gumbel_r', 'logistic', 'weibull_min'])
    def test_anderson_fit_params(self, distname):
        rng = np.random.default_rng(330691555377792039)
        real_distname = 'gumbel_l' if distname in {'extreme1', 'gumbel'} else distname
        dist = getattr(stats, real_distname)
        params = distcont[real_distname]
        x = dist.rvs(*params, size=1000, random_state=rng)
        res = stats.anderson(x, distname)
        assert res.fit_result.success

    def test_anderson_weibull_As(self):
        m = 1
        assert_equal(_get_As_weibull(1 / m), _Avals_weibull[-1])
        m = np.inf
        assert_equal(_get_As_weibull(1 / m), _Avals_weibull[0])