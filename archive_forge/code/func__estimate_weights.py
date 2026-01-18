import math
from numbers import Real
import numpy as np
from scipy.special import betaln, digamma, gammaln
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ._base import BaseMixture, _check_shape
from ._gaussian_mixture import (
def _estimate_weights(self, nk):
    """Estimate the parameters of the Dirichlet distribution.

        Parameters
        ----------
        nk : array-like of shape (n_components,)
        """
    if self.weight_concentration_prior_type == 'dirichlet_process':
        self.weight_concentration_ = (1.0 + nk, self.weight_concentration_prior_ + np.hstack((np.cumsum(nk[::-1])[-2::-1], 0)))
    else:
        self.weight_concentration_ = self.weight_concentration_prior_ + nk