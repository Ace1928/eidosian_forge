import warnings
from numbers import Real
import numpy as np
from scipy.special import xlogy
from ..exceptions import UndefinedMetricWarning
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
def _mean_tweedie_deviance(y_true, y_pred, sample_weight, power):
    """Mean Tweedie deviance regression loss."""
    p = power
    if p < 0:
        dev = 2 * (np.power(np.maximum(y_true, 0), 2 - p) / ((1 - p) * (2 - p)) - y_true * np.power(y_pred, 1 - p) / (1 - p) + np.power(y_pred, 2 - p) / (2 - p))
    elif p == 0:
        dev = (y_true - y_pred) ** 2
    elif p == 1:
        dev = 2 * (xlogy(y_true, y_true / y_pred) - y_true + y_pred)
    elif p == 2:
        dev = 2 * (np.log(y_pred / y_true) + y_true / y_pred - 1)
    else:
        dev = 2 * (np.power(y_true, 2 - p) / ((1 - p) * (2 - p)) - y_true * np.power(y_pred, 1 - p) / (1 - p) + np.power(y_pred, 2 - p) / (2 - p))
    return np.average(dev, weights=sample_weight)