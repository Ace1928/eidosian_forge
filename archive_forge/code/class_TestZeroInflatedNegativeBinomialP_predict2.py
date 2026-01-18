from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
class TestZeroInflatedNegativeBinomialP_predict2:

    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        cls.endog = np.asarray(data.endog)
        data.exog = np.asarray(data.exog)
        exog = data.exog
        start_params = np.array([-2.83983767, -2.31595924, -3.9263248, -4.01816431, -5.52251843, -2.4351714, -4.61636366, -4.17959785, -0.12960256, -0.05653484, -0.21206673, 0.08782572, -0.02991995, 0.22901208, 0.0620983, 0.06809681, 0.0841814, 0.185506, 1.36527888])
        mod = sm.ZeroInflatedNegativeBinomialP(cls.endog, exog, exog_infl=exog, p=2)
        res = mod.fit(start_params=start_params, method='bfgs', maxiter=1000, disp=False)
        cls.res = res

    def test_mean(self):
        assert_allclose(self.res.predict().mean(), self.endog.mean(), atol=0.02)

    def test_zero_nonzero_mean(self):
        mean1 = self.endog.mean()
        mean2 = (1 - self.res.predict(which='prob-zero').mean()) * self.res.predict(which='mean-nonzero').mean()
        assert_allclose(mean1, mean2, atol=0.2)