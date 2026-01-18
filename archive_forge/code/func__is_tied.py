import warnings
import numpy as np
from scipy import optimize
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from .descriptive import _OptFuncts
def _is_tied(self, endog, censors):
    """
        Indicated if an observation takes the same value as the next
        ordered observation.

        Parameters
        ----------
        endog : ndarray
            Models endogenous variable
        censors : ndarray
            arrat indicating a censored array

        Returns
        -------
        indic_ties : ndarray
            ties[i]=1 if endog[i]==endog[i+1] and
            censors[i]=censors[i+1]
        """
    nobs = int(self.nobs)
    endog_idx = endog[np.arange(nobs - 1)] == endog[np.arange(nobs - 1) + 1]
    censors_idx = censors[np.arange(nobs - 1)] == censors[np.arange(nobs - 1) + 1]
    indic_ties = endog_idx * censors_idx
    return np.int_(indic_ties)