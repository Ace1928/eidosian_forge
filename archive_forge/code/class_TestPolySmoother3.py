import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.sandbox.nonparametric import smoothers
from statsmodels.regression.linear_model import OLS, WLS
class TestPolySmoother3(BasePolySmoother, CheckSmoother):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        y, x, exog = (cls.y, cls.x, cls.exog)
        nobs = y.shape[0]
        weights = np.ones(nobs)
        weights[:nobs // 3] = 0.1
        weights[-nobs // 5:] = 2
        pmod = smoothers.PolySmoother(2, x)
        pmod.fit(y, weights=weights)
        cls.res_ps = pmod
        cls.res2 = WLS(y, exog[:, :2 + 1], weights=weights).fit()