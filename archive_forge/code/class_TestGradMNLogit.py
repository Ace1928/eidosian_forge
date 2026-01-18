import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import statsmodels.api as sm
from statsmodels.tools import numdiff
from statsmodels.tools.numdiff import (
class TestGradMNLogit(CheckGradLoglikeMixin):

    @classmethod
    def setup_class(cls):
        data = sm.datasets.anes96.load()
        data.exog = np.asarray(data.exog)
        data.endog = np.asarray(data.endog)
        exog = data.exog
        exog = sm.add_constant(exog, prepend=False)
        cls.mod = sm.MNLogit(data.endog, exog)
        res = cls.mod.fit(disp=0)
        cls.params = [res.params.ravel('F')]

    def test_hess(self):
        for test_params in self.params:
            he = self.mod.hessian(test_params)
            hefd = numdiff.approx_fprime_cs(test_params, self.mod.score)
            assert_almost_equal(he, hefd, decimal=DEC8)
            assert_almost_equal(he, hefd, decimal=7)
            hefd = numdiff.approx_fprime(test_params, self.mod.score, centered=True)
            assert_almost_equal(he, hefd, decimal=4)
            hefd = numdiff.approx_fprime(test_params, self.mod.score, 1e-09, centered=False)
            assert_almost_equal(he, hefd, decimal=2)
            hescs = numdiff.approx_fprime_cs(test_params, self.mod.score)
            assert_almost_equal(he, hescs, decimal=DEC8)
            hecs = numdiff.approx_hess_cs(test_params, self.mod.loglike)
            assert_almost_equal(he, hecs, decimal=5)
            hecs = numdiff.approx_hess3(test_params, self.mod.loglike, 0.0001)
            assert_almost_equal(he, hecs, decimal=0)