import numpy as np
from numpy.testing import assert_allclose, assert_equal
import statsmodels.base._penalties as smpen
from statsmodels.tools.numdiff import approx_fprime, approx_hess
class TestL2Constraints0(CheckPenalty):

    @classmethod
    def setup_class(cls):
        x0 = np.linspace(-0.2, 0.2, 11)
        cls.params = np.column_stack((x0, x0))
        cls.pen = smpen.L2ConstraintsPenalty()

    def test_equivalence(self):
        pen = self.pen
        x = self.params
        k = x.shape[1]
        pen2 = smpen.L2ConstraintsPenalty(weights=np.ones(k))
        pen3 = smpen.L2ConstraintsPenalty(restriction=np.eye(k))
        f = pen.func(x.T)
        d = pen.deriv(x.T)
        d2 = np.array([pen.deriv2(np.atleast_1d(xi)) for xi in x])
        for pen_ in [pen2, pen3]:
            assert_allclose(pen_.func(x.T), f, rtol=1e-07, atol=1e-08)
            assert_allclose(pen_.deriv(x.T), d, rtol=1e-07, atol=1e-08)
            d2_ = np.array([pen.deriv2(np.atleast_1d(xi)) for xi in x])
            assert_allclose(d2_, d2, rtol=1e-10, atol=1e-08)