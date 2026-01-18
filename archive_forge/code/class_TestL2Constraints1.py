import numpy as np
from numpy.testing import assert_allclose, assert_equal
import statsmodels.base._penalties as smpen
from statsmodels.tools.numdiff import approx_fprime, approx_hess
class TestL2Constraints1(CheckPenalty):

    @classmethod
    def setup_class(cls):
        x0 = np.linspace(-0.2, 0.2, 11)
        cls.params = np.column_stack((x0, x0))
        cls.pen = smpen.L2ConstraintsPenalty(restriction=[[1, 0], [1, 1]])

    def test_values(self):
        pen = self.pen
        x = self.params
        r = pen.restriction
        f = (r.dot(x.T) ** 2).sum(0)
        assert_allclose(pen.func(x.T), f, rtol=1e-07, atol=1e-08)