import math
from numbers import Real
import numpy as np
from scipy.special import betaln, digamma, gammaln
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ._base import BaseMixture, _check_shape
from ._gaussian_mixture import (
def _check_precision_parameters(self, X):
    """Check the prior parameters of the precision distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
    _, n_features = X.shape
    if self.degrees_of_freedom_prior is None:
        self.degrees_of_freedom_prior_ = n_features
    elif self.degrees_of_freedom_prior > n_features - 1.0:
        self.degrees_of_freedom_prior_ = self.degrees_of_freedom_prior
    else:
        raise ValueError("The parameter 'degrees_of_freedom_prior' should be greater than %d, but got %.3f." % (n_features - 1, self.degrees_of_freedom_prior))