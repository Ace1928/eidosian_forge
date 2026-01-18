import math
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from time import time
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, issparse
from .._loss.loss import (
from ..base import ClassifierMixin, RegressorMixin, _fit_context, is_classifier
from ..dummy import DummyClassifier, DummyRegressor
from ..exceptions import NotFittedError
from ..model_selection import train_test_split
from ..preprocessing import LabelEncoder
from ..tree import DecisionTreeRegressor
from ..tree._tree import DOUBLE, DTYPE, TREE_LEAF
from ..utils import check_array, check_random_state, column_or_1d
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.stats import _weighted_percentile
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import BaseEnsemble
from ._gradient_boosting import _random_sample_mask, predict_stage, predict_stages
def _init_raw_predictions(X, estimator, loss, use_predict_proba):
    """Return the initial raw predictions.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The data array.
    estimator : object
        The estimator to use to compute the predictions.
    loss : BaseLoss
        An instance of a loss function class.
    use_predict_proba : bool
        Whether estimator.predict_proba is used instead of estimator.predict.

    Returns
    -------
    raw_predictions : ndarray of shape (n_samples, K)
        The initial raw predictions. K is equal to 1 for binary
        classification and regression, and equal to the number of classes
        for multiclass classification. ``raw_predictions`` is casted
        into float64.
    """
    if use_predict_proba:
        predictions = estimator.predict_proba(X)
        if not loss.is_multiclass:
            predictions = predictions[:, 1]
        eps = np.finfo(np.float32).eps
        predictions = np.clip(predictions, eps, 1 - eps, dtype=np.float64)
    else:
        predictions = estimator.predict(X).astype(np.float64)
    if predictions.ndim == 1:
        return loss.link.link(predictions).reshape(-1, 1)
    else:
        return loss.link.link(predictions)