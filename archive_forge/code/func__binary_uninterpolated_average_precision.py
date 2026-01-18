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
def _binary_uninterpolated_average_precision(y_true, y_score, pos_label=1, sample_weight=None):
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])