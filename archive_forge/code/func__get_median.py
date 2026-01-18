import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from ..utils.fixes import _sparse_min_max, _sparse_nan_min_max
from ..utils.validation import _check_sample_weight
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
def _get_median(data, n_zeros):
    """Compute the median of data with n_zeros additional zeros.

    This function is used to support sparse matrices; it modifies data
    in-place.
    """
    n_elems = len(data) + n_zeros
    if not n_elems:
        return np.nan
    n_negative = np.count_nonzero(data < 0)
    middle, is_odd = divmod(n_elems, 2)
    data.sort()
    if is_odd:
        return _get_elem_at_rank(middle, data, n_negative, n_zeros)
    return (_get_elem_at_rank(middle - 1, data, n_negative, n_zeros) + _get_elem_at_rank(middle, data, n_negative, n_zeros)) / 2.0