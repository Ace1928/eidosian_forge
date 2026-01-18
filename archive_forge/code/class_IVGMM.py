from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
class IVGMM(GMM):
    """
    Basic class for instrumental variables estimation using GMM

    A linear function for the conditional mean is defined as default but the
    methods should be overwritten by subclasses, currently `LinearIVGMM` and
    `NonlinearIVGMM` are implemented as subclasses.

    See Also
    --------
    LinearIVGMM
    NonlinearIVGMM

    """
    results_class = 'IVGMMResults'

    def fitstart(self):
        """Create array of zeros"""
        return np.zeros(self.exog.shape[1])

    def start_weights(self, inv=True):
        """Starting weights"""
        zz = np.dot(self.instrument.T, self.instrument)
        nobs = self.instrument.shape[0]
        if inv:
            return zz / nobs
        else:
            return np.linalg.pinv(zz / nobs)

    def get_error(self, params):
        """Get error at params"""
        return self.endog - self.predict(params)

    def predict(self, params, exog=None):
        """Get prediction at params"""
        if exog is None:
            exog = self.exog
        return np.dot(exog, params)

    def momcond(self, params):
        """Error times instrument"""
        instrument = self.instrument
        return instrument * self.get_error(params)[:, None]