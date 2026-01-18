import warnings
from functools import partial
from numbers import Integral, Real
import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.stats import rankdata
from ..exceptions import UndefinedMetricWarning
from ..preprocessing import label_binarize
from ..utils import (
from ..utils._encode import _encode, _unique
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import stable_cumsum
from ..utils.fixes import trapezoid
from ..utils.multiclass import type_of_target
from ..utils.sparsefuncs import count_nonzero
from ..utils.validation import _check_pos_label_consistency, _check_sample_weight
from ._base import _average_binary_score, _average_multiclass_ovo_score
def _check_dcg_target_type(y_true):
    y_type = type_of_target(y_true, input_name='y_true')
    supported_fmt = ('multilabel-indicator', 'continuous-multioutput', 'multiclass-multioutput')
    if y_type not in supported_fmt:
        raise ValueError('Only {} formats are supported. Got {} instead'.format(supported_fmt, y_type))