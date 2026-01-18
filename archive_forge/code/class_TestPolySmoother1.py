import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.sandbox.nonparametric import smoothers
from statsmodels.regression.linear_model import OLS, WLS
class TestPolySmoother1(BasePolySmoother, CheckSmoother):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        y, x, exog = (cls.y, cls.x, cls.exog)
        pmod = smoothers.PolySmoother(2, x)
        pmod.fit(y)
        cls.res_ps = pmod
        cls.res2 = OLS(y, exog[:, :2 + 1]).fit()