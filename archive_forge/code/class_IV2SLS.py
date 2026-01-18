from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
class IV2SLS(LikelihoodModel):
    """
    Instrumental variables estimation using Two-Stage Least-Squares (2SLS)


    Parameters
    ----------
    endog : ndarray
       Endogenous variable, 1-dimensional or 2-dimensional array nobs by 1
    exog : ndarray
       Explanatory variables, 1-dimensional or 2-dimensional array nobs by k
    instrument : ndarray
       Instruments for explanatory variables. Must contain both exog
       variables that are not being instrumented and instruments

    Notes
    -----
    All variables in exog are instrumented in the calculations. If variables
    in exog are not supposed to be instrumented, then these variables
    must also to be included in the instrument array.

    Degrees of freedom in the calculation of the standard errors uses
    `df_resid = (nobs - k_vars)`.
    (This corresponds to the `small` option in Stata's ivreg2.)
    """

    def __init__(self, endog, exog, instrument=None):
        self.instrument, self.instrument_names = _ensure_2d(instrument, True)
        super().__init__(endog, exog)
        self.df_resid = self.exog.shape[0] - self.exog.shape[1]
        self.df_model = float(self.exog.shape[1] - self.k_constant)

    def initialize(self):
        self.wendog = self.endog
        self.wexog = self.exog

    def whiten(self, X):
        """Not implemented"""
        pass

    def fit(self):
        """estimate model using 2SLS IV regression

        Returns
        -------
        results : instance of RegressionResults
           regression result

        Notes
        -----
        This returns a generic RegressioResults instance as defined for the
        linear models.

        Parameter estimates and covariance are correct, but other results
        have not been tested yet, to see whether they apply without changes.

        """
        y, x, z = (self.endog, self.exog, self.instrument)
        ztz = np.dot(z.T, z)
        ztx = np.dot(z.T, x)
        self.xhatparams = xhatparams = np.linalg.solve(ztz, ztx)
        F = xhat = np.dot(z, xhatparams)
        FtF = np.dot(F.T, F)
        self.xhatprod = FtF
        Ftx = np.dot(F.T, x)
        Fty = np.dot(F.T, y)
        params = np.linalg.solve(FtF, Fty)
        Ftxinv = np.linalg.inv(Ftx)
        self.normalized_cov_params = np.dot(Ftxinv.T, np.dot(FtF, Ftxinv))
        lfit = IVRegressionResults(self, params, normalized_cov_params=self.normalized_cov_params)
        lfit.exog_hat_params = xhatparams
        lfit.exog_hat = xhat
        self._results_ols2nd = OLS(y, xhat).fit()
        return RegressionResultsWrapper(lfit)

    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        exog : array_like
            Design / exogenous data
        params : array_like, optional after fit has been called
            Parameters of a linear model

        Returns
        -------
        An array of fitted values

        Notes
        -----
        If the model as not yet been fit, params is not optional.
        """
        if exog is None:
            exog = self.exog
        return np.dot(exog, params)