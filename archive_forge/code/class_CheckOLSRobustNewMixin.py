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
class CheckOLSRobustNewMixin:

    def test_compare(self):
        rtol = getattr(self, 'rtol', 1e-10)
        assert_allclose(self.cov_robust, self.cov_robust2, rtol=rtol)
        assert_allclose(self.bse_robust, self.bse_robust2, rtol=rtol)

    def test_fvalue(self):
        if not getattr(self, 'skip_f', False):
            rtol = getattr(self, 'rtol', 1e-10)
            assert_allclose(self.res1.fvalue, self.res2.F, rtol=rtol)
            if hasattr(self.res2, 'Fp'):
                assert_allclose(self.res1.f_pvalue, self.res2.Fp, rtol=rtol)
        else:
            raise pytest.skip('TODO: document why this test is skipped')

    def test_confint(self):
        rtol = getattr(self, 'rtol', 1e-10)
        ci1 = self.res1.conf_int()
        ci2 = self.res2.params_table[:, 4:6]
        assert_allclose(ci1, ci2, rtol=rtol)
        crit1 = np.diff(ci1, 1).ravel() / 2 / self.res1.bse
        crit2 = np.diff(ci1, 1).ravel() / 2 / self.res1.bse
        assert_allclose(crit1, crit2, rtol=12)

    def test_ttest(self):
        res1 = self.res1
        res2 = self.res2
        rtol = getattr(self, 'rtol', 1e-10)
        rtolh = getattr(self, 'rtol', 1e-12)
        mat = np.eye(len(res1.params))
        tt = res1.t_test(mat, cov_p=self.cov_robust)
        assert_allclose(tt.effect, res2.params, rtol=rtolh)
        assert_allclose(tt.sd, res2.bse, rtol=rtol)
        assert_allclose(tt.tvalue, res2.tvalues, rtol=rtolh)
        assert_allclose(tt.pvalue, res2.pvalues, rtol=5 * rtol)
        ci1 = tt.conf_int()
        ci2 = self.res2.params_table[:, 4:6]
        assert_allclose(ci1, ci2, rtol=rtol)

    def test_scale(self):
        res1 = self.res1
        res2 = self.res2
        rtol = 1e-05
        skip = False
        if hasattr(res2, 'rss'):
            scale = res2.rss / (res2.N - res2.df_m - 1)
        elif hasattr(res2, 'rmse'):
            scale = res2.rmse ** 2
        else:
            skip = True
        if isinstance(res1.model, WLS):
            skip = True
        if not skip:
            assert_allclose(res1.scale, scale, rtol=rtol)
        if not res2.vcetype == 'Newey-West':
            r2 = res2.r2 if hasattr(res2, 'r2') else res2.r2c
            assert_allclose(res1.rsquared, r2, rtol=rtol, err_msg=str(skip))
        df_resid = res1.nobs - res1.df_model - 1
        assert_equal(res1.df_resid, df_resid)
        psum = (res1.resid_pearson ** 2).sum()
        assert_allclose(psum, df_resid, rtol=1e-13)

    @pytest.mark.smoke
    def test_summary(self):
        self.res1.summary()