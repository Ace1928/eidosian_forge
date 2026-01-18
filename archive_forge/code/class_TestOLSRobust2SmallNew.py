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
class TestOLSRobust2SmallNew(TestOLSRobust1, CheckOLSRobustNewMixin):

    def setup_method(self):
        res_ols = self.res1.get_robustcov_results('HC1', use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        self.bse_robust2 = res_ols.HC1_se
        self.cov_robust2 = res_ols.cov_HC1
        self.small = True
        self.res2 = res.results_ivhc0_small

    def test_compare(self):
        res1 = self.res1
        endog = res1.model.endog
        exog = res1.model.exog[:, [0, 2]]
        res_ols2 = OLS(endog, exog).fit()
        r_pval = 0.0307306938402991
        r_chi2 = 4.667944083588736
        r_df = 1
        assert_warns(InvalidTestWarning, res1.compare_lr_test, res_ols2)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            chi2, pval, df = res1.compare_lr_test(res_ols2)
        assert_allclose(chi2, r_chi2, rtol=1e-11)
        assert_allclose(pval, r_pval, rtol=1e-11)
        assert_equal(df, r_df)
        assert_warns(InvalidTestWarning, res1.compare_f_test, res_ols2)