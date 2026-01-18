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
class TestOLSRobustCluster2Fit(CheckOLSRobustCluster, CheckOLSRobustNewMixin):

    def setup_method(self):
        res_ols = self.res1.model.fit(cov_type='cluster', cov_kwds=dict(groups=self.groups, use_correction=True, use_t=True))
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster(self.res1, self.groups, use_correction=True)
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1
        self.small = True
        self.res2 = res2.results_cluster
        self.rtol = 1e-06
        self.rtolh = 1e-10

    def test_basic_inference(self):
        res1 = self.res1
        res2 = self.res2
        rtol = 1e-07
        assert_allclose(res1.params, res2.params, rtol=1e-08)
        assert_allclose(res1.bse, res2.bse, rtol=rtol)
        assert_allclose(res1.pvalues, res2.pvalues, rtol=rtol, atol=1e-20)
        ci = res2.params_table[:, 4:6]
        assert_allclose(res1.conf_int(), ci, rtol=5e-07, atol=1e-20)