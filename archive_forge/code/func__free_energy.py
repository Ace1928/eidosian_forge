import time
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy.special import expit  # logistic function
from ..base import (
from ..utils import check_random_state, gen_even_slices
from ..utils._param_validation import Interval
from ..utils.extmath import safe_sparse_dot
from ..utils.validation import check_is_fitted
def _free_energy(self, v):
    """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        free_energy : ndarray of shape (n_samples,)
            The value of the free energy.
        """
    return -safe_sparse_dot(v, self.intercept_visible_) - np.logaddexp(0, safe_sparse_dot(v, self.components_.T) + self.intercept_hidden_).sum(axis=1)