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
def _mean_hiddens(self, v):
    """Computes the probabilities P(h=1|v).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        """
    p = safe_sparse_dot(v, self.components_.T)
    p += self.intercept_hidden_
    return expit(p, out=p)