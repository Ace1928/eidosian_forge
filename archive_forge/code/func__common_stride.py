import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
def _common_stride(offsets, counts, itemsize):
    """
    Returns the stride between the fields, or None if the stride is not
    constant. The values in "counts" designate the lengths of
    subarrays. Subarrays are treated as many contiguous fields, with
    always positive stride.
    """
    if len(offsets) <= 1:
        return itemsize
    negative = offsets[1] < offsets[0]
    if negative:
        it = zip(reversed(offsets), reversed(counts))
    else:
        it = zip(offsets, counts)
    prev_offset = None
    stride = None
    for offset, count in it:
        if count != 1:
            if negative:
                return None
            if stride is None:
                stride = itemsize
            if stride != itemsize:
                return None
            end_offset = offset + (count - 1) * itemsize
        else:
            end_offset = offset
        if prev_offset is not None:
            new_stride = offset - prev_offset
            if stride is None:
                stride = new_stride
            if stride != new_stride:
                return None
        prev_offset = end_offset
    if negative:
        return -stride
    return stride