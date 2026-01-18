import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
@staticmethod
def _apply_missing_value(data, missing_value):
    """
        Applies the given missing value to the data array.

        Returns a numpy.ma array, with any value equal to missing_value masked
        out (unless missing_value is None, in which case the original array is
        returned).
        """
    if missing_value is None:
        newdata = data
    else:
        try:
            missing_value_isnan = np.isnan(missing_value)
        except (TypeError, NotImplementedError):
            missing_value_isnan = False
        if missing_value_isnan:
            mymask = np.isnan(data)
        else:
            mymask = data == missing_value
        newdata = np.ma.masked_where(mymask, data)
    return newdata