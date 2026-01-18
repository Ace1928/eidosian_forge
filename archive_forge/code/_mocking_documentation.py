import numpy as np
from ..base import BaseEstimator, ClassifierMixin
from ..utils._metadata_requests import RequestMethod
from .metaestimators import available_if
from .validation import _check_sample_weight, _num_samples, check_array, check_is_fitted
Fake score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Y : array-like of shape (n_samples, n_output) or (n_samples,)
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
            Either 0 or 1 depending of `foo_param` (i.e. `foo_param > 1 =>
            score=1` otherwise `score=0`).
        