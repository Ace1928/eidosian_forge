import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
from numpy.testing import (assert_array_less, assert_almost_equal,
class TestMyParetoRestriction(CheckGenericMixin):

    @classmethod
    def setup_class(cls):
        params = [2, 0, 2]
        nobs = 50
        np.random.seed(1234)
        rvs = stats.pareto.rvs(*params, **dict(size=nobs))
        mod_par = MyPareto(rvs)
        fixdf = np.nan * np.ones(3)
        fixdf[1] = -0.1
        mod_par.fixed_params = fixdf
        mod_par.fixed_paramsmask = np.isnan(fixdf)
        mod_par.start_params = mod_par.start_params[mod_par.fixed_paramsmask]
        mod_par.df_model = 0
        mod_par.k_extra = k_extra = 2
        mod_par.df_resid = mod_par.endog.shape[0] - mod_par.df_model - k_extra
        mod_par.data.xnames = ['shape', 'scale']
        cls.mod = mod_par
        cls.res1 = mod_par.fit(disp=None)
        cls.k_extra = k_extra
        cls.skip_bsejac = False