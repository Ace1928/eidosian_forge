import math
from numbers import Real
import numpy as np
from scipy.special import betaln, digamma, gammaln
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ._base import BaseMixture, _check_shape
from ._gaussian_mixture import (
def _check_means_parameters(self, X):
    """Check the parameters of the Gaussian distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
    _, n_features = X.shape
    if self.mean_precision_prior is None:
        self.mean_precision_prior_ = 1.0
    else:
        self.mean_precision_prior_ = self.mean_precision_prior
    if self.mean_prior is None:
        self.mean_prior_ = X.mean(axis=0)
    else:
        self.mean_prior_ = check_array(self.mean_prior, dtype=[np.float64, np.float32], ensure_2d=False)
        _check_shape(self.mean_prior_, (n_features,), 'means')