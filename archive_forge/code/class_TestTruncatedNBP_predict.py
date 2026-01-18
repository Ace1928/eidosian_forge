import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels import datasets
from statsmodels.tools.tools import add_constant
from statsmodels.tools.testing import Holder
from statsmodels.tools.sm_exceptions import (
from statsmodels.distributions.discrete import (
from statsmodels.discrete.truncated_model import (
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results.results_discrete import RandHIE
from .results import results_truncated as results_t
from .results import results_truncated_st as results_ts
class TestTruncatedNBP_predict:

    @classmethod
    def setup_class(cls):
        cls.expected_params = [1, 0.5, 0.5]
        np.random.seed(1234)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs // 2, 1] = 2
        mu_true = np.exp(exog.dot(cls.expected_params[:-1]))
        cls.endog = truncatednegbin.rvs(mu_true, cls.expected_params[-1], 2, 0, size=mu_true.shape)
        model = TruncatedLFNegativeBinomialP(cls.endog, exog, truncation=0, p=2)
        cls.res = model.fit(method='nm', maxiter=5000, maxfun=5000)

    def test_mean(self):
        assert_allclose(self.res.predict().mean(), self.endog.mean(), atol=0.2, rtol=0.2)

    def test_var(self):
        v = self.res.predict(which='var').mean()
        assert_allclose(v, self.endog.var(), atol=0.1, rtol=0.01)
        return
        assert_allclose(self.res.predict().mean() * self.res._dispersion_factor.mean(), self.endog.var(), atol=0.05, rtol=0.05)

    def test_predict_prob(self):
        res = self.res
        pr = res.predict(which='prob')
        pr2 = truncatednegbin.pmf(np.arange(29), res.predict(which='mean-main')[:, None], res.params[-1], 2, 0)
        assert_allclose(pr, pr2, rtol=1e-10, atol=1e-10)