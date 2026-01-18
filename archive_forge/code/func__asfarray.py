from numbers import Number
import operator
import os
import threading
import contextlib
import numpy as np
from .pypocketfft import good_size
def _asfarray(x):
    """
    Convert to array with floating or complex dtype.

    float16 values are also promoted to float32.
    """
    if not hasattr(x, 'dtype'):
        x = np.asarray(x)
    if x.dtype == np.float16:
        return np.asarray(x, np.float32)
    elif x.dtype.kind not in 'fc':
        return np.asarray(x, np.float64)
    dtype = x.dtype.newbyteorder('=')
    copy = not x.flags['ALIGNED']
    return np.array(x, dtype=dtype, copy=copy)