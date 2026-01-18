from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
class TestGeneralizedPoisson_underdispersion:

    @classmethod
    def setup_class(cls):
        cls.expected_params = [1, -0.5, -0.05]
        np.random.seed(1234)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs // 2, 1] = 2
        mu_true = np.exp(exog.dot(cls.expected_params[:-1]))
        cls.endog = sm.distributions.genpoisson_p.rvs(mu_true, cls.expected_params[-1], 1, size=len(mu_true))
        model_gp = sm.GeneralizedPoisson(cls.endog, exog, p=1)
        cls.res = model_gp.fit(method='nm', xtol=1e-06, maxiter=5000, maxfun=5000, disp=0)

    def test_basic(self):
        res = self.res
        endog = res.model.endog
        assert_allclose(endog.mean(), 1.42, rtol=0.001)
        assert_allclose(endog.var(), 1.2836, rtol=0.001)
        assert_allclose(res.params, self.expected_params, atol=0.07, rtol=0.1)
        assert_(res.mle_retvals['converged'] is True)
        assert_allclose(res.mle_retvals['fopt'], 1.418753161722015, rtol=0.01)

    def test_newton(self):
        res = self.res
        res2 = res.model.fit(start_params=res.params, method='newton', disp=0)
        assert_allclose(res.model.score(res.params), np.zeros(len(res2.params)), atol=0.01)
        assert_allclose(res.model.score(res2.params), np.zeros(len(res2.params)), atol=1e-10)
        assert_allclose(res.params, res2.params, atol=0.0001)

    def test_mean_var(self):
        assert_allclose(self.res.predict().mean(), self.endog.mean(), atol=0.1, rtol=0.1)
        assert_allclose(self.res.predict().mean() * self.res._dispersion_factor.mean(), self.endog.var(), atol=0.2, rtol=0.2)

    def test_predict_prob(self):
        res = self.res
        endog = res.model.endog
        freq = np.bincount(endog.astype(int))
        pr = res.predict(which='prob')
        pr2 = sm.distributions.genpoisson_p.pmf(np.arange(6)[:, None], res.predict(), res.params[-1], 1).T
        assert_allclose(pr, pr2, rtol=1e-10, atol=1e-10)
        expected = pr.sum(0)
        expected[-1] += pr.shape[0] - expected.sum()
        assert_allclose(freq.sum(), expected.sum(), rtol=1e-13)
        from scipy import stats
        chi2 = stats.chisquare(freq, expected)
        assert_allclose(chi2[:], (0.5511787456691261, 0.9901293016678583), rtol=0.01)

    def test_jac(self):
        check_jac(self, res=self.res)

    def test_distr(self):
        check_distr(self.res)