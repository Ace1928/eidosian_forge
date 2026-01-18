import warnings
from inspect import signature
from math import log
from numbers import Integral, Real
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.utils import Bunch
from ._loss import HalfBinomialLoss
from .base import (
from .isotonic import IsotonicRegression
from .model_selection import check_cv, cross_val_predict
from .preprocessing import LabelEncoder, label_binarize
from .svm import LinearSVC
from .utils import (
from .utils._param_validation import (
from .utils._plotting import _BinaryClassifierCurveDisplayMixin
from .utils._response import _get_response_values, _process_predict_proba
from .utils.metadata_routing import (
from .utils.multiclass import check_classification_targets
from .utils.parallel import Parallel, delayed
from .utils.validation import (
def _fit_classifier_calibrator_pair(estimator, X, y, train, test, method, classes, sample_weight=None, fit_params=None):
    """Fit a classifier/calibration pair on a given train/test split.

    Fit the classifier on the train set, compute its predictions on the test
    set and use the predictions as input to fit the calibrator along with the
    test labels.

    Parameters
    ----------
    estimator : estimator instance
        Cloned base estimator.

    X : array-like, shape (n_samples, n_features)
        Sample data.

    y : array-like, shape (n_samples,)
        Targets.

    train : ndarray, shape (n_train_indices,)
        Indices of the training subset.

    test : ndarray, shape (n_test_indices,)
        Indices of the testing subset.

    method : {'sigmoid', 'isotonic'}
        Method to use for calibration.

    classes : ndarray, shape (n_classes,)
        The target classes.

    sample_weight : array-like, default=None
        Sample weights for `X`.

    fit_params : dict, default=None
        Parameters to pass to the `fit` method of the underlying
        classifier.

    Returns
    -------
    calibrated_classifier : _CalibratedClassifier instance
    """
    fit_params_train = _check_method_params(X, params=fit_params, indices=train)
    X_train, y_train = (_safe_indexing(X, train), _safe_indexing(y, train))
    X_test, y_test = (_safe_indexing(X, test), _safe_indexing(y, test))
    estimator.fit(X_train, y_train, **fit_params_train)
    predictions, _ = _get_response_values(estimator, X_test, response_method=['decision_function', 'predict_proba'])
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    sw_test = None if sample_weight is None else _safe_indexing(sample_weight, test)
    calibrated_classifier = _fit_calibrator(estimator, predictions, y_test, classes, method, sample_weight=sw_test)
    return calibrated_classifier