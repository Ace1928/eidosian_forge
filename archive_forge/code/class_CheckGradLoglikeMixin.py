import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import statsmodels.api as sm
from statsmodels.tools import numdiff
from statsmodels.tools.numdiff import (
class CheckGradLoglikeMixin:

    def test_score(self):
        for test_params in self.params:
            sc = self.mod.score(test_params)
            scfd = numdiff.approx_fprime(test_params.ravel(), self.mod.loglike)
            assert_almost_equal(sc, scfd, decimal=1)
            sccs = numdiff.approx_fprime_cs(test_params.ravel(), self.mod.loglike)
            assert_almost_equal(sc, sccs, decimal=11)

    def test_hess(self):
        for test_params in self.params:
            he = self.mod.hessian(test_params)
            hefd = numdiff.approx_fprime_cs(test_params, self.mod.score)
            assert_almost_equal(he, hefd, decimal=DEC8)
            assert_almost_equal(he, hefd, decimal=7)
            hefd = numdiff.approx_fprime(test_params, self.mod.score, centered=True)
            assert_allclose(he, hefd, rtol=1e-09)
            hefd = numdiff.approx_fprime(test_params, self.mod.score, centered=False)
            assert_almost_equal(he, hefd, decimal=4)
            hescs = numdiff.approx_fprime_cs(test_params.ravel(), self.mod.score)
            assert_allclose(he, hescs, rtol=1e-13)
            hecs = numdiff.approx_hess_cs(test_params.ravel(), self.mod.loglike)
            assert_allclose(he, hecs, rtol=1e-09)
            grad = self.mod.score(test_params)
            hecs, gradcs = numdiff.approx_hess1(test_params, self.mod.loglike, 1e-06, return_grad=True)
            assert_almost_equal(he, hecs, decimal=1)
            assert_almost_equal(grad, gradcs, decimal=1)
            hecs, gradcs = numdiff.approx_hess2(test_params, self.mod.loglike, 0.0001, return_grad=True)
            assert_almost_equal(he, hecs, decimal=3)
            assert_almost_equal(grad, gradcs, decimal=1)
            hecs = numdiff.approx_hess3(test_params, self.mod.loglike, 1e-05)
            assert_almost_equal(he, hecs, decimal=4)