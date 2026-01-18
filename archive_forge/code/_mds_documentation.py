import warnings
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from ..base import BaseEstimator, _fit_context
from ..isotonic import IsotonicRegression
from ..metrics import euclidean_distances
from ..utils import check_array, check_random_state, check_symmetric
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.parallel import Parallel, delayed

        Fit the data from `X`, and returns the embedded coordinates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or                 (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.

        y : Ignored
            Not used, present for API consistency by convention.

        init : ndarray of shape (n_samples, n_components), default=None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            X transformed in the new space.
        