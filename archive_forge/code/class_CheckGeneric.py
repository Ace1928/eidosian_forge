from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
class CheckGeneric(CheckModelMixin):

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params, atol=1e-05, rtol=1e-05)

    def test_llf(self):
        assert_allclose(self.res1.llf, self.res2.llf, atol=1e-05, rtol=1e-05)

    def test_conf_int(self):
        assert_allclose(self.res1.conf_int(), self.res2.conf_int, atol=0.001, rtol=1e-05)

    def test_bse(self):
        assert_allclose(self.res1.bse, self.res2.bse, atol=0.001, rtol=0.001)

    def test_aic(self):
        assert_allclose(self.res1.aic, self.res2.aic, atol=0.01, rtol=0.01)

    def test_bic(self):
        assert_allclose(self.res1.aic, self.res2.aic, atol=0.1, rtol=0.1)

    def test_t(self):
        unit_matrix = np.identity(self.res1.params.size)
        t_test = self.res1.t_test(unit_matrix)
        assert_allclose(self.res1.tvalues, t_test.tvalue)

    def test_fit_regularized(self):
        model = self.res1.model
        alpha = np.ones(len(self.res1.params))
        alpha[-2:] = 0
        res_reg = model.fit_regularized(alpha=alpha * 0.01, disp=False, maxiter=500)
        assert_allclose(res_reg.params[2:], self.res1.params[2:], atol=0.05, rtol=0.05)

    def test_init_keys(self):
        init_kwds = self.res1.model._get_init_kwds()
        assert_equal(set(init_kwds.keys()), set(self.init_keys))
        for key, value in self.init_kwds.items():
            assert_equal(init_kwds[key], value)

    def test_null(self):
        self.res1.llnull
        exog_null = self.res1.res_null.model.exog
        exog_infl_null = self.res1.res_null.model.exog_infl
        assert_array_equal(exog_infl_null.shape, (len(self.res1.model.exog), 1))
        assert_equal(np.ptp(exog_null), 0)
        assert_equal(np.ptp(exog_infl_null), 0)

    @pytest.mark.smoke
    def test_summary(self):
        summ = self.res1.summary()
        assert 'Covariance Type:' in str(summ)