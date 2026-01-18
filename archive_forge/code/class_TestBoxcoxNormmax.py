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
class TestBoxcoxNormmax:

    def setup_method(self):
        self.x = _old_loggamma_rvs(5, size=50, random_state=12345) + 5

    def test_pearsonr(self):
        maxlog = stats.boxcox_normmax(self.x)
        assert_allclose(maxlog, 1.804465, rtol=1e-06)

    def test_mle(self):
        maxlog = stats.boxcox_normmax(self.x, method='mle')
        assert_allclose(maxlog, 1.758101, rtol=1e-06)
        _, maxlog_boxcox = stats.boxcox(self.x)
        assert_allclose(maxlog_boxcox, maxlog)

    def test_all(self):
        maxlog_all = stats.boxcox_normmax(self.x, method='all')
        assert_allclose(maxlog_all, [1.804465, 1.758101], rtol=1e-06)

    @pytest.mark.parametrize('method', ['mle', 'pearsonr', 'all'])
    @pytest.mark.parametrize('bounds', [(-1, 1), (1.1, 2), (-2, -1.1)])
    def test_bounded_optimizer_within_bounds(self, method, bounds):

        def optimizer(fun):
            return optimize.minimize_scalar(fun, bounds=bounds, method='bounded')
        maxlog = stats.boxcox_normmax(self.x, method=method, optimizer=optimizer)
        assert np.all(bounds[0] < maxlog)
        assert np.all(maxlog < bounds[1])

    def test_user_defined_optimizer(self):
        lmbda = stats.boxcox_normmax(self.x)
        lmbda_rounded = np.round(lmbda, 5)
        lmbda_range = np.linspace(lmbda_rounded - 0.01, lmbda_rounded + 0.01, 1001)

        class MyResult:
            pass

        def optimizer(fun):
            objs = []
            for lmbda in lmbda_range:
                objs.append(fun(lmbda))
            res = MyResult()
            res.x = lmbda_range[np.argmin(objs)]
            return res
        lmbda2 = stats.boxcox_normmax(self.x, optimizer=optimizer)
        assert lmbda2 != lmbda
        assert_allclose(lmbda2, lmbda, 1e-05)

    def test_user_defined_optimizer_and_brack_raises_error(self):
        optimizer = optimize.minimize_scalar
        stats.boxcox_normmax(self.x, brack=None, optimizer=optimizer)
        with pytest.raises(ValueError, match='`brack` must be None if `optimizer` is given'):
            stats.boxcox_normmax(self.x, brack=(-2.0, 2.0), optimizer=optimizer)

    @pytest.mark.parametrize('x', ([2003.0, 1950.0, 1997.0, 2000.0, 2009.0], [0.50000471, 0.50004979, 0.50005902, 0.50009312, 0.50001632]))
    def test_overflow(self, x):
        message = 'The optimal lambda is...'
        with pytest.warns(UserWarning, match=message):
            lmbda = stats.boxcox_normmax(x, method='mle')
        assert np.isfinite(special.boxcox(x, lmbda)).all()
        ymax = np.finfo(np.float64).max / 10000
        x_treme = np.max(x) if lmbda > 0 else np.min(x)
        y_extreme = special.boxcox(x_treme, lmbda)
        assert_allclose(y_extreme, ymax * np.sign(lmbda))