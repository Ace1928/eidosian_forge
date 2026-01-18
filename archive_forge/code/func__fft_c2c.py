import functools
import math
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._cache import get_plan_cache
def _fft_c2c(a, direction, norm, axes, overwrite_x, plan=None):
    for axis in axes:
        a = _exec_fft(a, direction, 'C2C', norm, axis, overwrite_x, plan=plan)
    return a