import functools
import itertools
import operator
import warnings
from . import numeric as _nx
from . import overrides
from .multiarray import array, asanyarray, normalize_axis_index
from . import fromnumeric as _from_nx
def _block_check_depths_match(arrays, parent_index=[]):
    """
    Recursive function checking that the depths of nested lists in `arrays`
    all match. Mismatch raises a ValueError as described in the block
    docstring below.

    The entire index (rather than just the depth) needs to be calculated
    for each innermost list, in case an error needs to be raised, so that
    the index of the offending list can be printed as part of the error.

    Parameters
    ----------
    arrays : nested list of arrays
        The arrays to check
    parent_index : list of int
        The full index of `arrays` within the nested lists passed to
        `_block_check_depths_match` at the top of the recursion.

    Returns
    -------
    first_index : list of int
        The full index of an element from the bottom of the nesting in
        `arrays`. If any element at the bottom is an empty list, this will
        refer to it, and the last index along the empty axis will be None.
    max_arr_ndim : int
        The maximum of the ndims of the arrays nested in `arrays`.
    final_size: int
        The number of elements in the final array. This is used the motivate
        the choice of algorithm used using benchmarking wisdom.

    """
    if type(arrays) is tuple:
        raise TypeError('{} is a tuple. Only lists can be used to arrange blocks, and np.block does not allow implicit conversion from tuple to ndarray.'.format(_block_format_index(parent_index)))
    elif type(arrays) is list and len(arrays) > 0:
        idxs_ndims = (_block_check_depths_match(arr, parent_index + [i]) for i, arr in enumerate(arrays))
        first_index, max_arr_ndim, final_size = next(idxs_ndims)
        for index, ndim, size in idxs_ndims:
            final_size += size
            if ndim > max_arr_ndim:
                max_arr_ndim = ndim
            if len(index) != len(first_index):
                raise ValueError('List depths are mismatched. First element was at depth {}, but there is an element at depth {} ({})'.format(len(first_index), len(index), _block_format_index(index)))
            if index[-1] is None:
                first_index = index
        return (first_index, max_arr_ndim, final_size)
    elif type(arrays) is list and len(arrays) == 0:
        return (parent_index + [None], 0, 0)
    else:
        size = _size(arrays)
        return (parent_index, _ndim(arrays), size)