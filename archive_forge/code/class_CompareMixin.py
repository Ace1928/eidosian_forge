import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
import statsmodels.api as sm
from statsmodels.miscmodels.count import PoissonGMLE, PoissonOffsetGMLE, \
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.tools.sm_exceptions import ValueWarning
class CompareMixin:

    def test_params(self):
        assert_almost_equal(self.res.params, self.res_glm.params, DEC5)
        assert_almost_equal(self.res.params, self.res_discrete.params, DEC5)

    def test_cov_params(self):
        assert_almost_equal(self.res.bse, self.res_glm.bse, DEC5)
        assert_almost_equal(self.res.bse, self.res_discrete.bse, DEC5)
        assert_almost_equal(self.res.tvalues, self.res_discrete.tvalues, DEC4)
        assert_almost_equal(self.res.pvalues, self.res_discrete.pvalues, DEC)

    def test_ttest(self):
        tt = self.res.t_test(np.eye(len(self.res.params)))
        from scipy import stats
        pvalue = stats.norm.sf(np.abs(tt.tvalue)) * 2
        assert_almost_equal(tt.tvalue, self.res.tvalues, DEC)
        assert_almost_equal(pvalue, self.res.pvalues, DEC)

    def test_df(self):
        res = self.res
        k_extra = getattr(self, 'k_extra', 0)
        nobs, k_vars = res.model.exog.shape
        assert res.df_resid == nobs - k_vars - k_extra
        assert res.df_model == k_vars - 1
        assert len(res.params) == k_vars + k_extra

    @pytest.mark.smoke
    def test_summary(self):
        self.res.summary()