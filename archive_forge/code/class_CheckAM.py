from statsmodels.compat.python import lrange
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats
import pytest
from statsmodels.sandbox.gam import AdditiveModel
from statsmodels.sandbox.gam import Model as GAM #?
from statsmodels.genmod.families import family, links
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS
class CheckAM:

    def test_predict(self):
        assert_almost_equal(self.res1.y_pred, self.res2.y_pred, decimal=2)
        assert_almost_equal(self.res1.y_predshort, self.res2.y_pred[:10], decimal=2)

    @pytest.mark.xfail(reason='Unknown, results do not match expected', raises=AssertionError, strict=True)
    def test_fitted(self):
        assert_almost_equal(self.res1.y_pred, self.res2.fittedvalues, decimal=2)
        assert_almost_equal(self.res1.y_predshort, self.res2.fittedvalues[:10], decimal=2)

    def test_params(self):
        assert_almost_equal(self.res1.params[1:], self.res2.params[1:], decimal=2)
        assert_almost_equal(self.res1.params[1], self.res2.params[1], decimal=2)

    @pytest.mark.xfail(reason='res_ps attribute does not exist', strict=True, raises=AttributeError)
    def test_df(self):
        assert_equal(self.res_ps.df_model(), self.res2.df_model)
        assert_equal(self.res_ps.df_fit(), self.res2.df_model)
        assert_equal(self.res_ps.df_resid(), self.res2.df_resid)