import array
import contextlib
import enum
import struct
def _LowerBound(values, value, pred):
    """Implementation of C++ std::lower_bound() algorithm."""
    first, last = (0, len(values))
    count = last - first
    while count > 0:
        i = first
        step = count // 2
        i += step
        if pred(values[i], value):
            i += 1
            first = i
            count -= step + 1
        else:
            count = step
    return first