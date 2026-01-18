import warnings
from numbers import Real
import numpy as np
from scipy.special import xlogy
from ..exceptions import UndefinedMetricWarning
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
def _assemble_r2_explained_variance(numerator, denominator, n_outputs, multioutput, force_finite):
    """Common part used by explained variance score and :math:`R^2` score."""
    nonzero_denominator = denominator != 0
    if not force_finite:
        output_scores = 1 - numerator / denominator
    else:
        nonzero_numerator = numerator != 0
        output_scores = np.ones([n_outputs])
        valid_score = nonzero_denominator & nonzero_numerator
        output_scores[valid_score] = 1 - numerator[valid_score] / denominator[valid_score]
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_scores
        elif multioutput == 'uniform_average':
            avg_weights = None
        elif multioutput == 'variance_weighted':
            avg_weights = denominator
            if not np.any(nonzero_denominator):
                avg_weights = None
    else:
        avg_weights = multioutput
    return np.average(output_scores, weights=avg_weights)