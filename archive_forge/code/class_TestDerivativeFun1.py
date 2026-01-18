import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import statsmodels.api as sm
from statsmodels.tools import numdiff
from statsmodels.tools.numdiff import (
class TestDerivativeFun1(CheckDerivativeMixin):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        xkols = np.dot(np.linalg.pinv(cls.x), cls.y)
        cls.params = [np.array([1.0, 1.0, 1.0]), xkols]
        cls.args = (cls.y, cls.x)

    def fun(self):
        return fun1

    def gradtrue(self, params):
        y, x = (self.y, self.x)
        return -x * 2 * (y - np.dot(x, params))[:, None]

    def hesstrue(self, params):
        return None
        y, x = (self.y, self.x)
        return -x * 2 * (y - np.dot(x, params))[:, None]