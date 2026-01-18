import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
import statsmodels.api as sm
from statsmodels.miscmodels.count import PoissonGMLE, PoissonOffsetGMLE, \
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.tools.sm_exceptions import ValueWarning
class TestPoissonOffset(CompareMixin):

    @classmethod
    def setup_class(cls):
        np.random.seed(98765678)
        nobs = 200
        rvs = np.random.randn(nobs, 6)
        data_exog = rvs
        data_exog = sm.add_constant(data_exog, prepend=False)
        xbeta = 1 + 0.1 * rvs.sum(1)
        data_endog = np.random.poisson(np.exp(xbeta))
        mod_glm = sm.GLM(data_endog, data_exog, family=sm.families.Poisson())
        cls.res_glm = mod_glm.fit()
        cls.res_discrete = Poisson(data_endog, data_exog).fit(disp=0)
        offset = cls.res_discrete.params[0] * data_exog[:, 0]
        cls.res_discrete = Poisson(data_endog, data_exog[:, 1:], offset=offset).fit(disp=0)
        mod_glm = sm.GLM(data_endog, data_exog, family=sm.families.Poisson())
        cls.res_glm = mod_glm.fit()
        modo = PoissonOffsetGMLE(data_endog, data_exog[:, 1:], offset=offset)
        cls.res = modo.fit(start_params=0.9 * cls.res_discrete.params, method='bfgs', disp=0)

    def test_params(self):
        assert_almost_equal(self.res.params, self.res_glm.params[1:], DEC)
        assert_almost_equal(self.res.params, self.res_discrete.params, DEC)

    def test_cov_params(self):
        assert_almost_equal(self.res.bse, self.res_glm.bse[1:], DEC - 1)
        assert_almost_equal(self.res.bse, self.res_discrete.bse, DEC5)