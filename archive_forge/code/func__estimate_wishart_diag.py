import math
from numbers import Real
import numpy as np
from scipy.special import betaln, digamma, gammaln
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ._base import BaseMixture, _check_shape
from ._gaussian_mixture import (
def _estimate_wishart_diag(self, nk, xk, sk):
    """Estimate the diag Wishart distribution parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)

        sk : array-like of shape (n_components, n_features)
        """
    _, n_features = xk.shape
    self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk
    diff = xk - self.mean_prior_
    self.covariances_ = self.covariance_prior_ + nk[:, np.newaxis] * (sk + (self.mean_precision_prior_ / self.mean_precision_)[:, np.newaxis] * np.square(diff))
    self.covariances_ /= self.degrees_of_freedom_[:, np.newaxis]