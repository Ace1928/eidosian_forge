import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def _select_via_looping(input, labels, idxs, positions, find_min, find_min_positions, find_max, find_max_positions, find_median):
    """Internal helper routine for _select.

    With relatively few labels it is faster to call this function rather than
    using the implementation based on cupy.lexsort.
    """
    find_positions = find_min_positions or find_max_positions
    arrays = []
    position_arrays = []
    for i in idxs:
        label_idx = labels == i
        arrays.append(input[label_idx])
        if find_positions:
            position_arrays.append(positions[label_idx])
    result = []
    if find_min:
        result += [_get_values(arrays, cupy.min)]
    if find_min_positions:
        result += [_get_positions(arrays, position_arrays, cupy.argmin)]
    if find_max:
        result += [_get_values(arrays, cupy.max)]
    if find_max_positions:
        result += [_get_positions(arrays, position_arrays, cupy.argmax)]
    if find_median:
        result += [_get_values(arrays, cupy.median)]
    return result