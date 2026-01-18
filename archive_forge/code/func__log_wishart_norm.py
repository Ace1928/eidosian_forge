import math
from numbers import Real
import numpy as np
from scipy.special import betaln, digamma, gammaln
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ._base import BaseMixture, _check_shape
from ._gaussian_mixture import (
def _log_wishart_norm(degrees_of_freedom, log_det_precisions_chol, n_features):
    """Compute the log of the Wishart distribution normalization term.

    Parameters
    ----------
    degrees_of_freedom : array-like of shape (n_components,)
        The number of degrees of freedom on the covariance Wishart
        distributions.

    log_det_precision_chol : array-like of shape (n_components,)
         The determinant of the precision matrix for each component.

    n_features : int
        The number of features.

    Return
    ------
    log_wishart_norm : array-like of shape (n_components,)
        The log normalization of the Wishart distribution.
    """
    return -(degrees_of_freedom * log_det_precisions_chol + degrees_of_freedom * n_features * 0.5 * math.log(2.0) + np.sum(gammaln(0.5 * (degrees_of_freedom - np.arange(n_features)[:, np.newaxis])), 0))