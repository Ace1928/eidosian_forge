import functools
import itertools
import operator
import warnings
from . import numeric as _nx
from . import overrides
from .multiarray import array, asanyarray, normalize_axis_index
from . import fromnumeric as _from_nx
def _block_setup(arrays):
    """
    Returns
    (`arrays`, list_ndim, result_ndim, final_size)
    """
    bottom_index, arr_ndim, final_size = _block_check_depths_match(arrays)
    list_ndim = len(bottom_index)
    if bottom_index and bottom_index[-1] is None:
        raise ValueError('List at {} cannot be empty'.format(_block_format_index(bottom_index)))
    result_ndim = max(arr_ndim, list_ndim)
    return (arrays, list_ndim, result_ndim, final_size)