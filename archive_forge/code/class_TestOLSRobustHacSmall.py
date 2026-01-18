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
class TestOLSRobustHacSmall(TestOLSRobust1):

    def setup_method(self):
        res_ols = self.res1
        cov1 = sw.cov_hac_simple(res_ols, nlags=4, use_correction=True)
        se1 = sw.se_cov(cov1)
        self.bse_robust = se1
        self.cov_robust = cov1
        self.small = True
        self.res2 = res.results_ivhac4_small