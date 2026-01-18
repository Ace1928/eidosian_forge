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
def _estimate_log_prob(self, X):
    """Estimate the log-probabilities log P(X | Z).

        Compute the log-probabilities per each component for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
        """
    pass