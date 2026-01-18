import numpy as np
from ...base import is_regressor
from ...preprocessing import LabelEncoder
from ...utils import _safe_indexing, check_matplotlib_support
from ...utils._response import _get_response_values
from ...utils.validation import (
def _check_boundary_response_method(estimator, response_method, class_of_interest):
    """Validate the response methods to be used with the fitted estimator.

    Parameters
    ----------
    estimator : object
        Fitted estimator to check.

    response_method : {'auto', 'predict_proba', 'decision_function', 'predict'}
        Specifies whether to use :term:`predict_proba`,
        :term:`decision_function`, :term:`predict` as the target response.
        If set to 'auto', the response method is tried in the following order:
        :term:`decision_function`, :term:`predict_proba`, :term:`predict`.

    class_of_interest : int, float, bool, str or None
        The class considered when plotting the decision. If the label is specified, it
        is then possible to plot the decision boundary in multiclass settings.

        .. versionadded:: 1.4

    Returns
    -------
    prediction_method : list of str or str
        The name or list of names of the response methods to use.
    """
    has_classes = hasattr(estimator, 'classes_')
    if has_classes and _is_arraylike_not_scalar(estimator.classes_[0]):
        msg = 'Multi-label and multi-output multi-class classifiers are not supported'
        raise ValueError(msg)
    if has_classes and len(estimator.classes_) > 2:
        if response_method not in {'auto', 'predict'} and class_of_interest is None:
            msg = "Multiclass classifiers are only supported when `response_method` is 'predict' or 'auto'. Else you must provide `class_of_interest` to plot the decision boundary of a specific class."
            raise ValueError(msg)
        prediction_method = 'predict' if response_method == 'auto' else response_method
    elif response_method == 'auto':
        if is_regressor(estimator):
            prediction_method = 'predict'
        else:
            prediction_method = ['decision_function', 'predict_proba', 'predict']
    else:
        prediction_method = response_method
    return prediction_method