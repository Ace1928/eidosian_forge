from numpy.ma import (
import numpy.ma as ma
import warnings
import numpy as np
from numpy import (
from numpy.core.records import (
def _guessvartypes(arr):
    """
    Tries to guess the dtypes of the str_ ndarray `arr`.

    Guesses by testing element-wise conversion. Returns a list of dtypes.
    The array is first converted to ndarray. If the array is 2D, the test
    is performed on the first line. An exception is raised if the file is
    3D or more.

    """
    vartypes = []
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[0]
    elif arr.ndim > 2:
        raise ValueError('The array should be 2D at most!')
    for f in arr:
        try:
            int(f)
        except (ValueError, TypeError):
            try:
                float(f)
            except (ValueError, TypeError):
                try:
                    complex(f)
                except (ValueError, TypeError):
                    vartypes.append(arr.dtype)
                else:
                    vartypes.append(np.dtype(complex))
            else:
                vartypes.append(np.dtype(float))
        else:
            vartypes.append(np.dtype(int))
    return vartypes