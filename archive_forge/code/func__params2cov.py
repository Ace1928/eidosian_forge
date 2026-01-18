import numpy as np
from scipy import linalg
from scipy.linalg import toeplitz
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.datasets import sunspots
from statsmodels.tsa.arima_process import (
def _params2cov(self, params, nobs):
    """get autocovariance matrix from ARMA regression parameter

        ar parameters are assumed to have rhs parameterization

        """
    ar = np.r_[[1], -params[:self.nar]]
    ma = np.r_[[1], params[-self.nma:]]
    autocov = arma_acovf(ar, ma, nobs=nobs)
    autocov = autocov[:nobs]
    sigma = toeplitz(autocov)
    return sigma