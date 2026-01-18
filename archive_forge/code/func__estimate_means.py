import math
from numbers import Real
import numpy as np
from scipy.special import betaln, digamma, gammaln
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ._base import BaseMixture, _check_shape
from ._gaussian_mixture import (
def _estimate_means(self, nk, xk):
    """Estimate the parameters of the Gaussian distribution.

        Parameters
        ----------
        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)
        """
    self.mean_precision_ = self.mean_precision_prior_ + nk
    self.means_ = (self.mean_precision_prior_ * self.mean_prior_ + nk[:, np.newaxis] * xk) / self.mean_precision_[:, np.newaxis]