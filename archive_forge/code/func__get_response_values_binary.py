import numpy as np
from ..base import is_classifier
from .multiclass import type_of_target
from .validation import _check_response_method, check_is_fitted
def _get_response_values_binary(estimator, X, response_method, pos_label=None):
    """Compute the response values of a binary classifier.

    Parameters
    ----------
    estimator : estimator instance
        Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
        in which the last estimator is a binary classifier.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.

    response_method : {'auto', 'predict_proba', 'decision_function'}
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. If set to 'auto',
        :term:`predict_proba` is tried first and if it does not exist
        :term:`decision_function` is tried next.

    pos_label : int, float, bool or str, default=None
        The class considered as the positive class when computing
        the metrics. By default, `estimators.classes_[1]` is
        considered as the positive class.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Target scores calculated from the provided response_method
        and pos_label.

    pos_label : int, float, bool or str
        The class considered as the positive class when computing
        the metrics.
    """
    classification_error = "Expected 'estimator' to be a binary classifier."
    check_is_fitted(estimator)
    if not is_classifier(estimator):
        raise ValueError(classification_error + f' Got {estimator.__class__.__name__} instead.')
    elif len(estimator.classes_) != 2:
        raise ValueError(classification_error + f' Got {len(estimator.classes_)} classes instead.')
    if response_method == 'auto':
        response_method = ['predict_proba', 'decision_function']
    return _get_response_values(estimator, X, response_method, pos_label=pos_label)