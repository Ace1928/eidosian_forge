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
def _check_zero_division(zero_division):
    if isinstance(zero_division, str) and zero_division == 'warn':
        return np.float64(0.0)
    elif isinstance(zero_division, (int, float)) and zero_division in [0, 1]:
        return np.float64(zero_division)
    else:
        return np.nan