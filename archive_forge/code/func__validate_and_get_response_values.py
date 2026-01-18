import numpy as np
from . import check_consistent_length, check_matplotlib_support
from ._response import _get_response_values_binary
from .multiclass import type_of_target
from .validation import _check_pos_label_consistency
@classmethod
def _validate_and_get_response_values(cls, estimator, X, y, *, response_method='auto', pos_label=None, name=None):
    check_matplotlib_support(f'{cls.__name__}.from_estimator')
    name = estimator.__class__.__name__ if name is None else name
    y_pred, pos_label = _get_response_values_binary(estimator, X, response_method=response_method, pos_label=pos_label)
    return (y_pred, pos_label, name)