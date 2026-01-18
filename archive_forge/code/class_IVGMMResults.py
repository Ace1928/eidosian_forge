from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
class IVGMMResults(GMMResults):
    """Results class of IVGMM"""

    @cache_readonly
    def fittedvalues(self):
        """Fitted values"""
        return self.model.predict(self.params)

    @cache_readonly
    def resid(self):
        """Residuals"""
        return self.model.endog - self.fittedvalues

    @cache_readonly
    def ssr(self):
        """Sum of square errors"""
        return (self.resid * self.resid).sum(0)