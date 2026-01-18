import warnings
import numpy as np
from scipy import optimize
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from .descriptive import _OptFuncts
class emplikeAFT:
    """

    Class for estimating and conducting inference in an AFT model.

    Parameters
    ----------

    endog: nx1 array
        Response variables that are subject to random censoring

    exog: nxk array
        Matrix of covariates

    censors: nx1 array
        array with entries 0 or 1.  0 indicates a response was
        censored.

    Attributes
    ----------
    nobs : float
        Number of observations
    endog : ndarray
        Endog attay
    exog : ndarray
        Exogenous variable matrix
    censors
        Censors array but sets the max(endog) to uncensored
    nvar : float
        Number of exogenous variables
    uncens_nobs : float
        Number of uncensored observations
    uncens_endog : ndarray
        Uncensored response variables
    uncens_exog : ndarray
        Exogenous variables of the uncensored observations

    Methods
    -------

    params:
        Fits model parameters

    test_beta:
        Tests if beta = b0 for any vector b0.

    Notes
    -----

    The data is immediately sorted in order of increasing endogenous
    variables

    The last observation is assumed to be uncensored which makes
    estimation and inference possible.
    """

    def __init__(self, endog, exog, censors):
        self.nobs = np.shape(exog)[0]
        self.endog = endog.reshape(self.nobs, 1)
        self.exog = exog.reshape(self.nobs, -1)
        self.censors = np.asarray(censors).reshape(self.nobs, 1)
        self.nvar = self.exog.shape[1]
        idx = np.lexsort((-self.censors[:, 0], self.endog[:, 0]))
        self.endog = self.endog[idx]
        self.exog = self.exog[idx]
        self.censors = self.censors[idx]
        self.censors[-1] = 1
        self.uncens_nobs = int(np.sum(self.censors))
        mask = self.censors.ravel().astype(bool)
        self.uncens_endog = self.endog[mask, :].reshape(-1, 1)
        self.uncens_exog = self.exog[mask, :]

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

    def _km_w_ties(self, tie_indic, untied_km):
        """
        Computes KM estimator value at each observation, taking into acocunt
        ties in the data.

        Parameters
        ----------
        tie_indic: 1d array
            Indicates if the i'th observation is the same as the ith +1
        untied_km: 1d array
            Km estimates at each observation assuming no ties.
        """
        num_same = 1
        idx_nums = []
        for obs_num in np.arange(int(self.nobs - 1))[::-1]:
            if tie_indic[obs_num] == 1:
                idx_nums.append(obs_num)
                num_same = num_same + 1
                untied_km[obs_num] = untied_km[obs_num + 1]
            elif tie_indic[obs_num] == 0 and num_same > 1:
                idx_nums.append(max(idx_nums) + 1)
                idx_nums = np.asarray(idx_nums)
                untied_km[idx_nums] = untied_km[idx_nums]
                num_same = 1
                idx_nums = []
        return untied_km.reshape(self.nobs, 1)

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

    def fit(self):
        """

        Fits an AFT model and returns results instance

        Parameters
        ----------
        None


        Returns
        -------
        Results instance.

        Notes
        -----
        To avoid dividing by zero, max(endog) is assumed to be uncensored.
        """
        return AFTResults(self)

    def predict(self, params, endog=None):
        if endog is None:
            endog = self.endog
        return np.dot(endog, params)