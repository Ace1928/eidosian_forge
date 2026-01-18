import contextlib
import functools
import operator
import warnings
import numpy as np
from numpy.core import overrides
def _unsigned_subtract(a, b):
    """
    Subtract two values where a >= b, and produce an unsigned result

    This is needed when finding the difference between the upper and lower
    bound of an int16 histogram
    """
    signed_to_unsigned = {np.byte: np.ubyte, np.short: np.ushort, np.intc: np.uintc, np.int_: np.uint, np.longlong: np.ulonglong}
    dt = np.result_type(a, b)
    try:
        dt = signed_to_unsigned[dt.type]
    except KeyError:
        return np.subtract(a, b, dtype=dt)
    else:
        return np.subtract(a, b, casting='unsafe', dtype=dt)