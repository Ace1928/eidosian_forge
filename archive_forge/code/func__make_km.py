import warnings
import numpy as np
from scipy import optimize
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from .descriptive import _OptFuncts
def _make_km(self, endog, censors):
    """

        Computes the Kaplan-Meier estimate for the weights in the AFT model

        Parameters
        ----------
        endog: nx1 array
            Array of response variables
        censors: nx1 array
            Censor-indicating variable

        Returns
        -------
        Kaplan Meier estimate for each observation

        Notes
        -----

        This function makes calls to _is_tied and km_w_ties to handle ties in
        the data.If a censored observation and an uncensored observation has
        the same value, it is assumed that the uncensored happened first.
        """
    nobs = self.nobs
    num = nobs - (np.arange(nobs) + 1.0)
    denom = nobs - (np.arange(nobs) + 1.0) + 1.0
    km = (num / denom).reshape(nobs, 1)
    km = km ** np.abs(censors - 1.0)
    km = np.cumprod(km)
    tied = self._is_tied(endog, censors)
    wtd_km = self._km_w_ties(tied, km)
    return (censors / wtd_km).reshape(nobs, 1)