import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import statsmodels.api as sm
from statsmodels.tools import numdiff
from statsmodels.tools.numdiff import (
class TestDerivativeFun2(CheckDerivativeMixin):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        xkols = np.dot(np.linalg.pinv(cls.x), cls.y)
        cls.params = [np.array([1.0, 1.0, 1.0]), xkols]
        cls.args = (cls.y, cls.x)

    def fun(self):
        return fun2

    def gradtrue(self, params):
        y, x = (self.y, self.x)
        return (-x * 2 * (y - np.dot(x, params))[:, None]).sum(0)

    def hesstrue(self, params):
        x = self.x
        return 2 * np.dot(x.T, x)