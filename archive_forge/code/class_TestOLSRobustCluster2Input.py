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
class TestOLSRobustCluster2Input(CheckOLSRobustCluster, CheckOLSRobustNewMixin):

    def setup_method(self):
        import pandas as pd
        fat_array = self.groups.reshape(-1, 1)
        fat_groups = pd.DataFrame(fat_array)
        res_ols = self.res1.get_robustcov_results('cluster', groups=fat_groups, use_correction=True, use_t=True)
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

    def test_too_many_groups(self):
        long_groups = self.groups.reshape(-1, 1)
        groups3 = np.hstack((long_groups, long_groups, long_groups))
        assert_raises(ValueError, self.res1.get_robustcov_results, 'cluster', groups=groups3, use_correction=True, use_t=True)

    def test_2way_dataframe(self):
        import pandas as pd
        long_groups = self.groups.reshape(-1, 1)
        groups2 = pd.DataFrame(np.hstack((long_groups, long_groups)))
        res = self.res1.get_robustcov_results('cluster', groups=groups2, use_correction=True, use_t=True)