import functools
import math
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._cache import get_plan_cache
def _get_fftn_out_size(in_shape, s, last_axis, value_type):
    if value_type == 'C2R':
        if s is None or s[-1] is None:
            out_size = 2 * (in_shape[last_axis] - 1)
        else:
            out_size = s[-1]
    elif value_type == 'R2C':
        out_size = in_shape[last_axis] // 2 + 1
    else:
        out_size = None
    return out_size