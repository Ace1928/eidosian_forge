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
Fit the model to the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        