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
class TestYeojohnson:

    def test_fixed_lmbda(self):
        rng = np.random.RandomState(12345)
        x = _old_loggamma_rvs(5, size=50, random_state=rng) + 5
        assert np.all(x > 0)
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt, x)
        xt = stats.yeojohnson(x, lmbda=-1)
        assert_allclose(xt, 1 - 1 / (x + 1))
        xt = stats.yeojohnson(x, lmbda=0)
        assert_allclose(xt, np.log(x + 1))
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt, x)
        x = _old_loggamma_rvs(5, size=50, random_state=rng) - 5
        assert np.all(x < 0)
        xt = stats.yeojohnson(x, lmbda=2)
        assert_allclose(xt, -np.log(-x + 1))
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt, x)
        xt = stats.yeojohnson(x, lmbda=3)
        assert_allclose(xt, 1 / (-x + 1) - 1)
        x = _old_loggamma_rvs(5, size=50, random_state=rng) - 2
        assert not np.all(x < 0)
        assert not np.all(x >= 0)
        pos = x >= 0
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt[pos], x[pos])
        xt = stats.yeojohnson(x, lmbda=-1)
        assert_allclose(xt[pos], 1 - 1 / (x[pos] + 1))
        xt = stats.yeojohnson(x, lmbda=0)
        assert_allclose(xt[pos], np.log(x[pos] + 1))
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt[pos], x[pos])
        neg = ~pos
        xt = stats.yeojohnson(x, lmbda=2)
        assert_allclose(xt[neg], -np.log(-x[neg] + 1))
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt[neg], x[neg])
        xt = stats.yeojohnson(x, lmbda=3)
        assert_allclose(xt[neg], 1 / (-x[neg] + 1) - 1)

    @pytest.mark.parametrize('lmbda', [0, 0.1, 0.5, 2])
    def test_lmbda_None(self, lmbda):

        def _inverse_transform(x, lmbda):
            x_inv = np.zeros(x.shape, dtype=x.dtype)
            pos = x >= 0
            if abs(lmbda) < np.spacing(1.0):
                x_inv[pos] = np.exp(x[pos]) - 1
            else:
                x_inv[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda) - 1
            if abs(lmbda - 2) > np.spacing(1.0):
                x_inv[~pos] = 1 - np.power(-(2 - lmbda) * x[~pos] + 1, 1 / (2 - lmbda))
            else:
                x_inv[~pos] = 1 - np.exp(-x[~pos])
            return x_inv
        n_samples = 20000
        np.random.seed(1234567)
        x = np.random.normal(loc=0, scale=1, size=n_samples)
        x_inv = _inverse_transform(x, lmbda)
        xt, maxlog = stats.yeojohnson(x_inv)
        assert_allclose(maxlog, lmbda, atol=0.01)
        assert_almost_equal(0, np.linalg.norm(x - xt) / n_samples, decimal=2)
        assert_almost_equal(0, xt.mean(), decimal=1)
        assert_almost_equal(1, xt.std(), decimal=1)

    def test_empty(self):
        assert_(stats.yeojohnson([]).shape == (0,))

    def test_array_like(self):
        x = stats.norm.rvs(size=100, loc=0, random_state=54321)
        xt1, _ = stats.yeojohnson(x)
        xt2, _ = stats.yeojohnson(list(x))
        assert_allclose(xt1, xt2, rtol=1e-12)

    @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
    def test_input_dtype_complex(self, dtype):
        x = np.arange(6, dtype=dtype)
        err_msg = 'Yeo-Johnson transformation is not defined for complex numbers.'
        with pytest.raises(ValueError, match=err_msg):
            stats.yeojohnson(x)

    @pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int16, np.int32])
    def test_input_dtype_integer(self, dtype):
        x_int = np.arange(8, dtype=dtype)
        x_float = np.arange(8, dtype=np.float64)
        xt_int, lmbda_int = stats.yeojohnson(x_int)
        xt_float, lmbda_float = stats.yeojohnson(x_float)
        assert_allclose(xt_int, xt_float, rtol=1e-07)
        assert_allclose(lmbda_int, lmbda_float, rtol=1e-07)

    def test_input_high_variance(self):
        x = np.array([3251637.22, 620695.44, 11642969.0, 2223468.22, 85307500.0, 16494389.89, 917215.88, 11642969.0, 2145773.87, 4962000.0, 620695.44, 651234.5, 1907876.71, 4053297.88, 3251637.22, 3259103.08, 9547969.0, 20631286.23, 12807072.08, 2383819.84, 90114500.0, 17209575.46, 12852969.0, 2414609.99, 2170368.23])
        xt_yeo, lam_yeo = stats.yeojohnson(x)
        xt_box, lam_box = stats.boxcox(x + 1)
        assert_allclose(xt_yeo, xt_box, rtol=1e-06)
        assert_allclose(lam_yeo, lam_box, rtol=1e-06)

    @pytest.mark.parametrize('x', [np.array([1.0, float('nan'), 2.0]), np.array([1.0, float('inf'), 2.0]), np.array([1.0, -float('inf'), 2.0]), np.array([-1.0, float('nan'), float('inf'), -float('inf'), 1.0])])
    def test_nonfinite_input(self, x):
        with pytest.raises(ValueError, match='Yeo-Johnson input must be finite'):
            xt_yeo, lam_yeo = stats.yeojohnson(x)

    @pytest.mark.parametrize('x', [np.array([2003.0, 1950.0, 1997.0, 2000.0, 2009.0, 2009.0, 1980.0, 1999.0, 2007.0, 1991.0]), np.array([2003.0, 1950.0, 1997.0, 2000.0, 2009.0]), np.array([2.003e+203, 1.95e+203, 1.997e+203, 2e+203, 2.009e+203])])
    def test_overflow(self, x):

        def optimizer(fun, lam_yeo):
            out = optimize.fminbound(fun, -lam_yeo, lam_yeo, xtol=1.48e-08)
            result = optimize.OptimizeResult()
            result.x = out
            return result
        with np.errstate(all='raise'):
            xt_yeo, lam_yeo = stats.yeojohnson(x)
            xt_box, lam_box = stats.boxcox(x + 1, optimizer=partial(optimizer, lam_yeo=lam_yeo))
            assert np.isfinite(np.var(xt_yeo))
            assert np.isfinite(np.var(xt_box))
            assert_allclose(lam_yeo, lam_box, rtol=1e-06)
            assert_allclose(xt_yeo, xt_box, rtol=0.0001)

    @pytest.mark.parametrize('x', [np.array([2003.0, 1950.0, 1997.0, 2000.0, 2009.0, 2009.0, 1980.0, 1999.0, 2007.0, 1991.0]), np.array([2003.0, 1950.0, 1997.0, 2000.0, 2009.0])])
    @pytest.mark.parametrize('scale', [1, 1e-12, 1e-32, 1e-150, 1e+32, 1e+200])
    @pytest.mark.parametrize('sign', [1, -1])
    def test_overflow_underflow_signed_data(self, x, scale, sign):
        with np.errstate(all='raise'):
            xt_yeo, lam_yeo = stats.yeojohnson(sign * x * scale)
            assert np.all(np.sign(sign * x) == np.sign(xt_yeo))
            assert np.isfinite(lam_yeo)
            assert np.isfinite(np.var(xt_yeo))

    @pytest.mark.parametrize('x', [np.array([0, 1, 2, 3]), np.array([0, -1, 2, -3]), np.array([0, 0, 0])])
    @pytest.mark.parametrize('sign', [1, -1])
    @pytest.mark.parametrize('brack', [None, (-2, 2)])
    def test_integer_signed_data(self, x, sign, brack):
        with np.errstate(all='raise'):
            x_int = sign * x
            x_float = x_int.astype(np.float64)
            lam_yeo_int = stats.yeojohnson_normmax(x_int, brack=brack)
            xt_yeo_int = stats.yeojohnson(x_int, lmbda=lam_yeo_int)
            lam_yeo_float = stats.yeojohnson_normmax(x_float, brack=brack)
            xt_yeo_float = stats.yeojohnson(x_float, lmbda=lam_yeo_float)
            assert np.all(np.sign(x_int) == np.sign(xt_yeo_int))
            assert np.isfinite(lam_yeo_int)
            assert np.isfinite(np.var(xt_yeo_int))
            assert lam_yeo_int == lam_yeo_float
            assert np.all(xt_yeo_int == xt_yeo_float)