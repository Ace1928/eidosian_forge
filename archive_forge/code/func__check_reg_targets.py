import warnings
from numbers import Real
import numpy as np
from scipy.special import xlogy
from ..exceptions import UndefinedMetricWarning
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
def _check_reg_targets(y_true, y_pred, multioutput, dtype='numeric'):
    """Check that y_true and y_pred belong to the same regression task.

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like

    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().

    dtype : str or list, default="numeric"
        the dtype argument passed to check_array.

    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'.

    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples, n_outputs)
        Estimated target values.

    multioutput : array-like of shape (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.
    """
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError('y_true and y_pred have different number of output ({0}!={1})'.format(y_true.shape[1], y_pred.shape[1]))
    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ('raw_values', 'uniform_average', 'variance_weighted')
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError("Allowed 'multioutput' string values are {}. You provided multioutput={!r}".format(allowed_multioutput_str, multioutput))
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError('Custom weights are useful only in multi-output cases.')
        elif n_outputs != len(multioutput):
            raise ValueError('There must be equally many custom weights (%d) as outputs (%d).' % (len(multioutput), n_outputs))
    y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'
    return (y_type, y_true, y_pred, multioutput)