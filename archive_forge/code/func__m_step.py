import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from time import time
import numpy as np
from scipy.special import logsumexp
from .. import cluster
from ..base import BaseEstimator, DensityMixin, _fit_context
from ..cluster import kmeans_plusplus
from ..exceptions import ConvergenceWarning
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.validation import check_is_fitted
@abstractmethod
def _m_step(self, X, log_resp):
    """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
    pass