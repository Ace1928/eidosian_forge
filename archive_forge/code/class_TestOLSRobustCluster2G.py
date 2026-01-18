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
class TestOLSRobustCluster2G(CheckOLSRobustCluster, CheckOLSRobustNewMixin):

    def setup_method(self):
        res_ols = self.res1.get_robustcov_results('cluster', groups=(self.groups, self.time), use_correction=True, use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster_2groups(self.res1, self.groups, group2=self.time, use_correction=True)[0]
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1
        self.small = True
        self.res2 = res2.results_cluster_2groups_small
        self.rtol = 0.35
        self.rtolh = 1e-10