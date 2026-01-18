import numbers
import numpy
import cupy
def _set_pad_area(padded, axis, width_pair, value_pair):
    """Set an empty-padded area in given dimension.
    """
    left_slice = _slice_at_axis(slice(None, width_pair[0]), axis)
    padded[left_slice] = value_pair[0]
    right_slice = _slice_at_axis(slice(padded.shape[axis] - width_pair[1], None), axis)
    padded[right_slice] = value_pair[1]