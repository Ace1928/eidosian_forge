import warnings
from numbers import Integral, Real
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import xlogy
from ..exceptions import UndefinedMetricWarning
from ..preprocessing import LabelBinarizer, LabelEncoder
from ..utils import (
from ..utils._array_api import _union1d, _weighted_sum, get_namespace
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.extmath import _nanaverage
from ..utils.multiclass import type_of_target, unique_labels
from ..utils.sparsefuncs import count_nonzero
from ..utils.validation import _check_pos_label_consistency, _num_samples
def _check_set_wise_labels(y_true, y_pred, average, labels, pos_label):
    """Validation associated with set-wise metrics.

    Returns identified labels.
    """
    average_options = (None, 'micro', 'macro', 'weighted', 'samples')
    if average not in average_options and average != 'binary':
        raise ValueError('average has to be one of ' + str(average_options))
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    present_labels = unique_labels(y_true, y_pred).tolist()
    if average == 'binary':
        if y_type == 'binary':
            if pos_label not in present_labels:
                if len(present_labels) >= 2:
                    raise ValueError(f'pos_label={pos_label} is not a valid label. It should be one of {present_labels}')
            labels = [pos_label]
        else:
            average_options = list(average_options)
            if y_type == 'multiclass':
                average_options.remove('samples')
            raise ValueError("Target is %s but average='binary'. Please choose another average setting, one of %r." % (y_type, average_options))
    elif pos_label not in (None, 1):
        warnings.warn("Note that pos_label (set to %r) is ignored when average != 'binary' (got %r). You may use labels=[pos_label] to specify a single positive class." % (pos_label, average), UserWarning)
    return labels