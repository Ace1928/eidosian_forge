import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.stats
import threadpoolctl
import sklearn
from ..externals._packaging.version import parse as parse_version
from .deprecation import deprecated
def _smallest_admissible_index_dtype(arrays=(), maxval=None, check_contents=False):
    """Based on input (integer) arrays `a`, determine a suitable index data
    type that can hold the data in the arrays.

    This function returns `np.int64` if it either required by `maxval` or based on the
    largest precision of the dtype of the arrays passed as argument, or by the their
    contents (when `check_contents is True`). If none of the condition requires
    `np.int64` then this function returns `np.int32`.

    Parameters
    ----------
    arrays : ndarray or tuple of ndarrays, default=()
        Input arrays whose types/contents to check.

    maxval : float, default=None
        Maximum value needed.

    check_contents : bool, default=False
        Whether to check the values in the arrays and not just their types.
        By default, check only the types.

    Returns
    -------
    dtype : {np.int32, np.int64}
        Suitable index data type (int32 or int64).
    """
    int32min = np.int32(np.iinfo(np.int32).min)
    int32max = np.int32(np.iinfo(np.int32).max)
    if maxval is not None:
        if maxval > np.iinfo(np.int64).max:
            raise ValueError(f'maxval={maxval} is to large to be represented as np.int64.')
        if maxval > int32max:
            return np.int64
    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)
    for arr in arrays:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f'Arrays should be of type np.ndarray, got {type(arr)} instead.')
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f'Array dtype {arr.dtype} is not supported for index dtype. We expect integral values.')
        if not np.can_cast(arr.dtype, np.int32):
            if not check_contents:
                return np.int64
            if arr.size == 0:
                continue
            else:
                maxval = arr.max()
                minval = arr.min()
                if minval < int32min or maxval > int32max:
                    return np.int64
    return np.int32