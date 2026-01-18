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
class TestOLSRobustCluster2LargeFit(CheckOLSRobustCluster, CheckOLSRobustNewMixin):

    def setup_method(self):
        model = OLS(self.res1.model.endog, self.res1.model.exog)
        res_ols = model.fit(cov_type='cluster', cov_kwds=dict(groups=self.groups, use_correction=False, use_t=False, df_correction=True))
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster(self.res1, self.groups, use_correction=False)
        se1 = sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1
        self.small = False
        self.res2 = res2.results_cluster_large
        self.skip_f = True
        self.rtol = 1e-06
        self.rtolh = 1e-10

    @pytest.mark.skip(reason='GH#1189 issuecomment-29141741')
    def test_fvalue(self):
        super().test_fvalue()