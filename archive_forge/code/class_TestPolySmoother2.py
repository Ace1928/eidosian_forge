import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.sandbox.nonparametric import smoothers
from statsmodels.regression.linear_model import OLS, WLS
class TestPolySmoother2(BasePolySmoother, CheckSmoother):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        y, x, exog = (cls.y, cls.x, cls.exog)
        pmod = smoothers.PolySmoother(3, x)
        pmod.smooth(y)
        cls.res_ps = pmod
        cls.res2 = OLS(y, exog[:, :3 + 1]).fit()