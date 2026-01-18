import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from ..utils.fixes import _sparse_min_max, _sparse_nan_min_max
from ..utils.validation import _check_sample_weight
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
def _raise_error_wrong_axis(axis):
    if axis not in (0, 1):
        raise ValueError('Unknown axis value: %d. Use 0 for rows, or 1 for columns' % axis)