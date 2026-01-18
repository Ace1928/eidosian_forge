import pytest
import numpy as np
from numpy.testing import assert_allclose
from statsmodels.tools.tools import add_constant
from statsmodels.tools.testing import Holder
from statsmodels.miscmodels.tmodel import TLinearModel
class CheckTLinearModelMixin:

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.params[:-2], res2.loc_fit.coefficients, atol=3e-05)
        assert_allclose(res1.bse[:-2], res2.loc_fit.table[:, 1], rtol=0.003, atol=1e-05)
        assert_allclose(res1.tvalues[:-2], res2.loc_fit.table[:, 2], rtol=0.003, atol=1e-05)
        assert_allclose(res1.pvalues[:-2], res2.loc_fit.table[:, 3], rtol=0.009, atol=1e-05)
        assert_allclose(res1.params[-2], res2.dof, rtol=5e-05)
        assert_allclose(res1.bse[-2], res2.dofse, rtol=0.16, atol=1e-05)
        scale_est = np.sqrt(res2.scale_fit.fitted_values.mean())
        assert_allclose(res1.params[-1], scale_est, atol=1e-05)
        assert_allclose(res1.llf, res2.logLik, atol=1e-05)

    def test_bse(self):
        res1 = self.res1
        assert_allclose(res1.bsejac, res1.bse, rtol=0.15, atol=0.002)
        assert_allclose(res1.bsejac, res1.bse, rtol=0.1, atol=0.004)

    def test_fitted(self):
        res1 = self.res1
        res2 = self.res2
        fittedvalues = res1.predict()
        resid = res1.model.endog - fittedvalues
        assert_allclose(fittedvalues, res2.loc_fit.fitted_values, rtol=0.00025)
        assert_allclose(resid, res2.loc_fit.residuals, atol=2e-06)

    def test_formula(self):
        res1 = self.res1
        resf = self.resf
        assert_allclose(res1.params, resf.params, atol=0.0001)
        assert_allclose(res1.bse, resf.bse, rtol=5e-05)
        assert_allclose(res1.model.endog, resf.model.endog, rtol=1e-10)
        assert_allclose(res1.model.exog, resf.model.exog, rtol=1e-10)

    def test_df(self):
        res = self.res1
        k_extra = getattr(self, 'k_extra', 0)
        nobs, k_vars = res.model.exog.shape
        assert res.df_resid == nobs - k_vars - k_extra
        assert res.df_model == k_vars - 1
        assert len(res.params) == k_vars + k_extra

    @pytest.mark.smoke
    def test_smoke(self):
        res1 = self.res1
        resf = self.resf
        contr = np.eye(len(res1.params))
        res1.summary()
        res1.t_test(contr)
        res1.f_test(contr)
        resf.summary()
        resf.t_test(contr)
        resf.f_test(contr)