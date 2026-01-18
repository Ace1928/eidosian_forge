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
class TestCircFuncs:

    @pytest.mark.parametrize('test_func,expected', [(stats.circmean, 0.167690146), (stats.circvar, 0.006455174270186603), (stats.circstd, 6.520702116)])
    def test_circfuncs(self, test_func, expected):
        x = np.array([355, 5, 2, 359, 10, 350])
        assert_allclose(test_func(x, high=360), expected, rtol=1e-07)

    def test_circfuncs_small(self):
        x = np.array([20, 21, 22, 18, 19, 20.5, 19.2])
        M1 = x.mean()
        M2 = stats.circmean(x, high=360)
        assert_allclose(M2, M1, rtol=1e-05)
        V1 = (x * np.pi / 180).var()
        V1 = V1 / 2.0
        V2 = stats.circvar(x, high=360)
        assert_allclose(V2, V1, rtol=0.0001)
        S1 = x.std()
        S2 = stats.circstd(x, high=360)
        assert_allclose(S2, S1, rtol=0.0001)

    @pytest.mark.parametrize('test_func, numpy_func', [(stats.circmean, np.mean), (stats.circvar, np.var), (stats.circstd, np.std)])
    def test_circfuncs_close(self, test_func, numpy_func):
        x = np.array([0.12675364631578953] * 10 + [0.12675365920187928] * 100)
        circstat = test_func(x)
        normal = numpy_func(x)
        assert_allclose(circstat, normal, atol=2e-08)

    def test_circmean_axis(self):
        x = np.array([[355, 5, 2, 359, 10, 350], [351, 7, 4, 352, 9, 349], [357, 9, 8, 358, 4, 356]])
        M1 = stats.circmean(x, high=360)
        M2 = stats.circmean(x.ravel(), high=360)
        assert_allclose(M1, M2, rtol=1e-14)
        M1 = stats.circmean(x, high=360, axis=1)
        M2 = [stats.circmean(x[i], high=360) for i in range(x.shape[0])]
        assert_allclose(M1, M2, rtol=1e-14)
        M1 = stats.circmean(x, high=360, axis=0)
        M2 = [stats.circmean(x[:, i], high=360) for i in range(x.shape[1])]
        assert_allclose(M1, M2, rtol=1e-14)

    def test_circvar_axis(self):
        x = np.array([[355, 5, 2, 359, 10, 350], [351, 7, 4, 352, 9, 349], [357, 9, 8, 358, 4, 356]])
        V1 = stats.circvar(x, high=360)
        V2 = stats.circvar(x.ravel(), high=360)
        assert_allclose(V1, V2, rtol=1e-11)
        V1 = stats.circvar(x, high=360, axis=1)
        V2 = [stats.circvar(x[i], high=360) for i in range(x.shape[0])]
        assert_allclose(V1, V2, rtol=1e-11)
        V1 = stats.circvar(x, high=360, axis=0)
        V2 = [stats.circvar(x[:, i], high=360) for i in range(x.shape[1])]
        assert_allclose(V1, V2, rtol=1e-11)

    def test_circstd_axis(self):
        x = np.array([[355, 5, 2, 359, 10, 350], [351, 7, 4, 352, 9, 349], [357, 9, 8, 358, 4, 356]])
        S1 = stats.circstd(x, high=360)
        S2 = stats.circstd(x.ravel(), high=360)
        assert_allclose(S1, S2, rtol=1e-11)
        S1 = stats.circstd(x, high=360, axis=1)
        S2 = [stats.circstd(x[i], high=360) for i in range(x.shape[0])]
        assert_allclose(S1, S2, rtol=1e-11)
        S1 = stats.circstd(x, high=360, axis=0)
        S2 = [stats.circstd(x[:, i], high=360) for i in range(x.shape[1])]
        assert_allclose(S1, S2, rtol=1e-11)

    @pytest.mark.parametrize('test_func,expected', [(stats.circmean, 0.167690146), (stats.circvar, 0.006455174270186603), (stats.circstd, 6.520702116)])
    def test_circfuncs_array_like(self, test_func, expected):
        x = [355, 5, 2, 359, 10, 350]
        assert_allclose(test_func(x, high=360), expected, rtol=1e-07)

    @pytest.mark.parametrize('test_func', [stats.circmean, stats.circvar, stats.circstd])
    def test_empty(self, test_func):
        assert_(np.isnan(test_func([])))

    @pytest.mark.parametrize('test_func', [stats.circmean, stats.circvar, stats.circstd])
    def test_nan_propagate(self, test_func):
        x = [355, 5, 2, 359, 10, 350, np.nan]
        assert_(np.isnan(test_func(x, high=360)))

    @pytest.mark.parametrize('test_func,expected', [(stats.circmean, {None: np.nan, 0: 355.66582264, 1: 0.28725053}), (stats.circvar, {None: np.nan, 0: 0.002570671054089924, 1: 0.005545914017677123}), (stats.circstd, {None: np.nan, 0: 4.11093193, 1: 6.04265394})])
    def test_nan_propagate_array(self, test_func, expected):
        x = np.array([[355, 5, 2, 359, 10, 350, 1], [351, 7, 4, 352, 9, 349, np.nan], [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
        for axis in expected.keys():
            out = test_func(x, high=360, axis=axis)
            if axis is None:
                assert_(np.isnan(out))
            else:
                assert_allclose(out[0], expected[axis], rtol=1e-07)
                assert_(np.isnan(out[1:]).all())

    @pytest.mark.parametrize('test_func,expected', [(stats.circmean, {None: 359.4178026893944, 0: np.array([353.0, 6.0, 3.0, 355.5, 9.5, 349.5]), 1: np.array([0.16769015, 358.66510252])}), (stats.circvar, {None: 0.008396678483192477, 0: np.array([1.9997969, 0.4999873, 0.4999873, 6.1230956, 0.1249992, 0.1249992]) * (np.pi / 180) ** 2, 1: np.array([0.006455174270186603, 0.01016767581393285])}), (stats.circstd, {None: 7.440570778057074, 0: np.array([2.00020313, 1.00002539, 1.00002539, 3.50108929, 0.50000317, 0.50000317]), 1: np.array([6.52070212, 8.19138093])})])
    def test_nan_omit_array(self, test_func, expected):
        x = np.array([[355, 5, 2, 359, 10, 350, np.nan], [351, 7, 4, 352, 9, 349, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
        for axis in expected.keys():
            out = test_func(x, high=360, nan_policy='omit', axis=axis)
            if axis is None:
                assert_allclose(out, expected[axis], rtol=1e-07)
            else:
                assert_allclose(out[:-1], expected[axis], rtol=1e-07)
                assert_(np.isnan(out[-1]))

    @pytest.mark.parametrize('test_func,expected', [(stats.circmean, 0.167690146), (stats.circvar, 0.006455174270186603), (stats.circstd, 6.520702116)])
    def test_nan_omit(self, test_func, expected):
        x = [355, 5, 2, 359, 10, 350, np.nan]
        assert_allclose(test_func(x, high=360, nan_policy='omit'), expected, rtol=1e-07)

    @pytest.mark.parametrize('test_func', [stats.circmean, stats.circvar, stats.circstd])
    def test_nan_omit_all(self, test_func):
        x = [np.nan, np.nan, np.nan, np.nan, np.nan]
        assert_(np.isnan(test_func(x, nan_policy='omit')))

    @pytest.mark.parametrize('test_func', [stats.circmean, stats.circvar, stats.circstd])
    def test_nan_omit_all_axis(self, test_func):
        x = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan]])
        out = test_func(x, nan_policy='omit', axis=1)
        assert_(np.isnan(out).all())
        assert_(len(out) == 2)

    @pytest.mark.parametrize('x', [[355, 5, 2, 359, 10, 350, np.nan], np.array([[355, 5, 2, 359, 10, 350, np.nan], [351, 7, 4, 352, np.nan, 9, 349]])])
    @pytest.mark.parametrize('test_func', [stats.circmean, stats.circvar, stats.circstd])
    def test_nan_raise(self, test_func, x):
        assert_raises(ValueError, test_func, x, high=360, nan_policy='raise')

    @pytest.mark.parametrize('x', [[355, 5, 2, 359, 10, 350, np.nan], np.array([[355, 5, 2, 359, 10, 350, np.nan], [351, 7, 4, 352, np.nan, 9, 349]])])
    @pytest.mark.parametrize('test_func', [stats.circmean, stats.circvar, stats.circstd])
    def test_bad_nan_policy(self, test_func, x):
        assert_raises(ValueError, test_func, x, high=360, nan_policy='foobar')

    def test_circmean_scalar(self):
        x = 1.0
        M1 = x
        M2 = stats.circmean(x)
        assert_allclose(M2, M1, rtol=1e-05)

    def test_circmean_range(self):
        m = stats.circmean(np.arange(0, 2, 0.1), np.pi, -np.pi)
        assert_(m < np.pi)
        assert_(m > -np.pi)

    def test_circfuncs_uint8(self):
        x = np.array([150, 10], dtype='uint8')
        assert_equal(stats.circmean(x, high=180), 170.0)
        assert_allclose(stats.circvar(x, high=180), 0.2339555554617, rtol=1e-07)
        assert_allclose(stats.circstd(x, high=180), 20.91551378, rtol=1e-07)