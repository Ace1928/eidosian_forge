import warnings
import numpy as np
from scipy import linalg
from .. import config_context
from ..base import BaseEstimator, _fit_context
from ..metrics.pairwise import pairwise_distances
from ..utils import check_array
from ..utils._param_validation import validate_params
from ..utils.extmath import fast_logdet
Compute the squared Mahalanobis distances of given observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The observations, the Mahalanobis distances of the which we
            compute. Observations are assumed to be drawn from the same
            distribution than the data used in fit.

        Returns
        -------
        dist : ndarray of shape (n_samples,)
            Squared Mahalanobis distances of the observations.
        