import numpy as np
from numpy.testing import (
import pytest
from scipy import stats
from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import OLS, WLS
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.sm_exceptions import InvalidTestWarning
from statsmodels.tools.tools import add_constant
from .results import (
class TestOLSRobust1(CheckOLSRobust):

    def setup_method(self):
        res_ols = self.res1
        self.bse_robust = res_ols.HC1_se
        self.cov_robust = res_ols.cov_HC1
        self.small = True
        self.res2 = res.results_hc0

    @classmethod
    def setup_class(cls):
        d2 = macrodata.load_pandas().data
        g_gdp = 400 * np.diff(np.log(d2['realgdp'].values))
        g_inv = 400 * np.diff(np.log(d2['realinv'].values))
        exogg = add_constant(np.c_[g_gdp, d2['realint'][:-1].values], prepend=False)
        cls.res1 = OLS(g_inv, exogg).fit()

    def test_qr_equiv(self):
        res2 = self.res1.model.fit(method='qr')
        assert_allclose(self.res1.HC0_se, res2.HC0_se)