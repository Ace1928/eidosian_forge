import numbers
import time
import warnings
from collections import Counter
from contextlib import suppress
from functools import partial
from numbers import Real
from traceback import format_exc
import numpy as np
import scipy.sparse as sp
from joblib import logger
from ..base import clone, is_classifier
from ..exceptions import FitFailedWarning, UnsetMetadataPassedError
from ..metrics import check_scoring, get_scorer_names
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from ..preprocessing import LabelEncoder
from ..utils import Bunch, _safe_indexing, check_random_state, indexable
from ..utils._param_validation import (
from ..utils.metadata_routing import (
from ..utils.metaestimators import _safe_split
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_method_params, _num_samples
from ._split import check_cv
def _fit_and_predict(estimator, X, y, train, test, fit_params, method):
    """Fit estimator and predict values for a given dataset split.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit.

        .. versionchanged:: 0.20
            X is only required to be an object with finite length or shape now

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    train : array-like of shape (n_train_samples,)
        Indices of training samples.

    test : array-like of shape (n_test_samples,)
        Indices of test samples.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    method : str
        Invokes the passed method name of the passed estimator.

    Returns
    -------
    predictions : sequence
        Result of calling 'estimator.method'
    """
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_method_params(X, params=fit_params, indices=train)
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, _ = _safe_split(estimator, X, y, test, train)
    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)
    func = getattr(estimator, method)
    predictions = func(X_test)
    encode = method in ['decision_function', 'predict_proba', 'predict_log_proba'] and y is not None
    if encode:
        if isinstance(predictions, list):
            predictions = [_enforce_prediction_order(estimator.classes_[i_label], predictions[i_label], n_classes=len(set(y[:, i_label])), method=method) for i_label in range(len(predictions))]
        else:
            n_classes = len(set(y)) if y.ndim == 1 else y.shape[1]
            predictions = _enforce_prediction_order(estimator.classes_, predictions, n_classes, method)
    return predictions