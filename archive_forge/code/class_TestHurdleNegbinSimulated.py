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
class TestHurdleNegbinSimulated(CheckHurdlePredict):

    @classmethod
    def setup_class(cls):
        nobs = 2000
        exog = np.column_stack((np.ones(nobs), np.linspace(0, 3, nobs)))
        y_fake = np.arange(nobs) // (nobs / 3)
        mod = HurdleCountModel(y_fake, exog, dist='negbin', zerodist='negbin')
        p_dgp = np.array([-0.4, 2, 0.5, 0.2, 0.5, 0.5])
        probs = mod.predict(p_dgp, which='prob', y_values=np.arange(50))
        cdf = probs.cumsum(1)
        n = cdf.shape[0]
        cdf = np.column_stack((cdf, np.ones(n)))
        rng = np.random.default_rng(987456348)
        u = rng.random((n, 1))
        endog = np.argmin(cdf < u, axis=1)
        mod_hnb = HurdleCountModel(endog, exog, dist='negbin', zerodist='negbin')
        cls.res1 = mod_hnb.fit(maxiter=300)
        df_null = 4
        cls.res2 = Holder(nobs=nobs, k_params=6, df_model=2, df_null=df_null, df_resid=nobs - 6, k_extra=df_null - 1, exog_names=['zm_const', 'zm_x1', 'zm_alpha', 'const', 'x1', 'alpha'])