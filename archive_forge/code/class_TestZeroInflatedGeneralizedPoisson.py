from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
@pytest.mark.slow
class TestZeroInflatedGeneralizedPoisson(CheckGeneric):

    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        cls.endog = np.asarray(data.endog)
        data.exog = np.asarray(data.exog)
        exog = sm.add_constant(data.exog[:, 1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog[:, 0], prepend=False)
        cls.res1 = sm.ZeroInflatedGeneralizedPoisson(data.endog, exog, exog_infl=exog_infl, p=1).fit(method='newton', maxiter=500, disp=False)
        cls.res1._results._attach_nullmodel = True
        cls.init_keys = ['exog_infl', 'exposure', 'inflation', 'offset', 'p']
        cls.init_kwds = {'inflation': 'logit', 'p': 1}
        res2 = RandHIE.zero_inflated_generalized_poisson
        cls.res2 = res2

    def test_bse(self):
        pass

    def test_conf_int(self):
        pass

    def test_bic(self):
        pass

    def test_t(self):
        unit_matrix = np.identity(self.res1.params.size)
        t_test = self.res1.t_test(unit_matrix)
        assert_allclose(self.res1.tvalues, t_test.tvalue)

    def test_minimize(self, reset_randomstate):
        model = self.res1.model
        start_params = self.res1.mle_settings['start_params']
        res_ncg = model.fit(start_params=start_params, method='minimize', min_method='trust-ncg', maxiter=500, disp=False)
        assert_allclose(res_ncg.params, self.res2.params, atol=0.001, rtol=0.04)
        assert_allclose(res_ncg.bse, self.res2.bse, atol=0.001, rtol=0.6)
        assert_(res_ncg.mle_retvals['converged'] is True)
        res_dog = model.fit(start_params=start_params, method='minimize', min_method='dogleg', maxiter=500, disp=False)
        assert_allclose(res_dog.params, self.res2.params, atol=0.001, rtol=0.003)
        assert_allclose(res_dog.bse, self.res2.bse, atol=0.001, rtol=0.6)
        assert_(res_dog.mle_retvals['converged'] is True)
        random_state = np.random.RandomState(1)
        seed = {'seed': random_state}
        res_bh = model.fit(start_params=start_params, method='basinhopping', niter=500, stepsize=0.1, niter_success=None, disp=False, interval=1, **seed)
        assert_allclose(res_bh.params, self.res2.params, atol=0.0001, rtol=0.0001)
        assert_allclose(res_bh.bse, self.res2.bse, atol=0.001, rtol=0.6)