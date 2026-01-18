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
def _warn_prf(average, modifier, msg_start, result_size):
    axis0, axis1 = ('sample', 'label')
    if average == 'samples':
        axis0, axis1 = (axis1, axis0)
    msg = '{0} ill-defined and being set to 0.0 {{0}} no {1} {2}s. Use `zero_division` parameter to control this behavior.'.format(msg_start, modifier, axis0)
    if result_size == 1:
        msg = msg.format('due to')
    else:
        msg = msg.format('in {0}s with'.format(axis1))
    warnings.warn(msg, UndefinedMetricWarning, stacklevel=2)