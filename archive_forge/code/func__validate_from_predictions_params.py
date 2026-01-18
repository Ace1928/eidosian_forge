import numpy as np
from . import check_consistent_length, check_matplotlib_support
from ._response import _get_response_values_binary
from .multiclass import type_of_target
from .validation import _check_pos_label_consistency
@classmethod
def _validate_from_predictions_params(cls, y_true, y_pred, *, sample_weight=None, pos_label=None, name=None):
    check_matplotlib_support(f'{cls.__name__}.from_predictions')
    if type_of_target(y_true) != 'binary':
        raise ValueError(f'The target y is not binary. Got {type_of_target(y_true)} type of target.')
    check_consistent_length(y_true, y_pred, sample_weight)
    pos_label = _check_pos_label_consistency(pos_label, y_true)
    name = name if name is not None else 'Classifier'
    return (pos_label, name)