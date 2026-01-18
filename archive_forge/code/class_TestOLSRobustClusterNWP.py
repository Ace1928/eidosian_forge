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
class TestOLSRobustClusterNWP(CheckOLSRobustCluster, CheckOLSRobustNewMixin):

    def setup_method(self):
        res_ols = self.res1.get_robustcov_results('nw-panel', time=self.time, maxlags=4, use_correction='hac', use_t=True, df_correction=False)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_nw_panel(self.res1, 4, self.tidx)
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1
        self.small = True
        self.res2 = res2.results_nw_panel4
        self.skip_f = True
        self.rtol = 1e-06
        self.rtolh = 1e-10

    def test_keyword(self):
        res_ols = self.res1.get_robustcov_results('hac-panel', time=self.time, maxlags=4, use_correction='hac', use_t=True, df_correction=False)
        assert_allclose(res_ols.bse, self.res1.bse, rtol=1e-12)