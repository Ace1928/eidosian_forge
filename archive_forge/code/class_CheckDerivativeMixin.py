import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import statsmodels.api as sm
from statsmodels.tools import numdiff
from statsmodels.tools.numdiff import (
class CheckDerivativeMixin:

    @classmethod
    def setup_class(cls):
        nobs = 200
        np.random.seed(187678)
        x = np.random.randn(nobs, 3)
        xk = np.array([1, 2, 3])
        xk = np.array([1.0, 1.0, 1.0])
        beta = xk
        y = np.dot(x, beta) + 0.1 * np.random.randn(nobs)
        xkols = np.dot(np.linalg.pinv(x), y)
        cls.x = x
        cls.y = y
        cls.params = [np.array([1.0, 1.0, 1.0]), xkols]
        cls.init()

    @classmethod
    def init(cls):
        pass

    def test_grad_fun1_fd(self):
        for test_params in self.params:
            gtrue = self.gradtrue(test_params)
            fun = self.fun()
            epsilon = 1e-06
            gfd = numdiff.approx_fprime(test_params, fun, epsilon=epsilon, args=self.args)
            gfd += numdiff.approx_fprime(test_params, fun, epsilon=-epsilon, args=self.args)
            gfd /= 2.0
            assert_almost_equal(gtrue, gfd, decimal=DEC6)

    def test_grad_fun1_fdc(self):
        for test_params in self.params:
            gtrue = self.gradtrue(test_params)
            fun = self.fun()
            gfd = numdiff.approx_fprime(test_params, fun, epsilon=1e-08, args=self.args, centered=True)
            assert_almost_equal(gtrue, gfd, decimal=DEC5)

    def test_grad_fun1_cs(self):
        for test_params in self.params:
            gtrue = self.gradtrue(test_params)
            fun = self.fun()
            gcs = numdiff.approx_fprime_cs(test_params, fun, args=self.args)
            assert_almost_equal(gtrue, gcs, decimal=DEC13)

    def test_hess_fun1_fd(self):
        for test_params in self.params:
            hetrue = self.hesstrue(test_params)
            if hetrue is not None:
                fun = self.fun()
                hefd = numdiff.approx_hess1(test_params, fun, args=self.args)
                assert_almost_equal(hetrue, hefd, decimal=DEC3)
                hefd = numdiff.approx_hess2(test_params, fun, args=self.args)
                assert_almost_equal(hetrue, hefd, decimal=DEC3)
                hefd = numdiff.approx_hess3(test_params, fun, args=self.args)
                assert_almost_equal(hetrue, hefd, decimal=DEC3)

    def test_hess_fun1_cs(self):
        for test_params in self.params:
            hetrue = self.hesstrue(test_params)
            if hetrue is not None:
                fun = self.fun()
                hecs = numdiff.approx_hess_cs(test_params, fun, args=self.args)
                assert_almost_equal(hetrue, hecs, decimal=DEC6)