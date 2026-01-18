import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
from numpy.testing import (assert_array_less, assert_almost_equal,
class TestMyPareto1(CheckGenericMixin):

    @classmethod
    def setup_class(cls):
        params = [2, 0, 2]
        nobs = 100
        np.random.seed(1234)
        rvs = stats.pareto.rvs(*params, **dict(size=nobs))
        mod_par = MyPareto(rvs)
        mod_par.fixed_params = None
        mod_par.fixed_paramsmask = None
        mod_par.df_model = 0
        mod_par.k_extra = k_extra = 3
        mod_par.df_resid = mod_par.endog.shape[0] - mod_par.df_model - k_extra
        mod_par.data.xnames = ['shape', 'loc', 'scale']
        cls.mod = mod_par
        cls.res1 = mod_par.fit(disp=None)
        cls.k_extra = k_extra
        cls.skip_bsejac = True

    def test_minsupport(self):
        params = self.res1.params
        x_min = self.res1.endog.min()
        p_min = params[1] + params[2]
        assert_array_less(p_min, x_min)
        assert_almost_equal(p_min, x_min, decimal=2)