from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
class TestZeroInflatedPoisson_predict:

    @classmethod
    def setup_class(cls):
        expected_params = [1, 0.5]
        np.random.seed(999)
        nobs = 2000
        exog = np.ones((nobs, 2))
        exog[:nobs // 2, 1] = 2
        mu_true = exog.dot(expected_params)
        cls.endog = sm.distributions.zipoisson.rvs(mu_true, 0.05, size=mu_true.shape)
        model = sm.ZeroInflatedPoisson(cls.endog, exog)
        cls.res = model.fit(method='bfgs', maxiter=5000, disp=False)
        cls.params_true = [mu_true, 0.05, nobs]

    def test_mean(self):

        def compute_conf_interval_95(mu, prob_infl, nobs):
            dispersion_factor = 1 + prob_infl * mu
            var = (dispersion_factor * (1 - prob_infl) * mu).mean()
            var += (((1 - prob_infl) * mu) ** 2).mean()
            var -= ((1 - prob_infl) * mu).mean() ** 2
            std = np.sqrt(var)
            conf_intv_95 = 2 * std / np.sqrt(nobs)
            return conf_intv_95
        conf_interval_95 = compute_conf_interval_95(*self.params_true)
        assert_allclose(self.res.predict().mean(), self.endog.mean(), atol=conf_interval_95, rtol=0)

    def test_var(self):

        def compute_mixture_var(dispersion_factor, prob_main, mu):
            prob_infl = 1 - prob_main
            var = (dispersion_factor * (1 - prob_infl) * mu).mean()
            var += (((1 - prob_infl) * mu) ** 2).mean()
            var -= ((1 - prob_infl) * mu).mean() ** 2
            return var
        res = self.res
        var_fitted = compute_mixture_var(res._dispersion_factor, res.predict(which='prob-main'), res.predict(which='mean-main'))
        assert_allclose(var_fitted.mean(), self.endog.var(), atol=0.05, rtol=0.05)

    def test_predict_prob(self):
        res = self.res
        pr = res.predict(which='prob')
        pr2 = sm.distributions.zipoisson.pmf(np.arange(pr.shape[1])[:, None], res.predict(), 0.05).T
        assert_allclose(pr, pr2, rtol=0.05, atol=0.05)

    def test_predict_options(self):
        res = self.res
        n = 5
        pr1 = res.predict(which='prob')
        pr0 = res.predict(exog=res.model.exog[:n], which='prob')
        assert_allclose(pr0, pr1[:n], rtol=1e-10)
        fitted1 = res.predict()
        fitted0 = res.predict(exog=res.model.exog[:n])
        assert_allclose(fitted0, fitted1[:n], rtol=1e-10)