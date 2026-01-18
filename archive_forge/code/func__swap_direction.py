import functools
import math
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._cache import get_plan_cache
def _swap_direction(norm):
    if norm in (None, 'backward'):
        norm = 'forward'
    elif norm == 'forward':
        norm = 'backward'
    elif norm != 'ortho':
        raise ValueError('Invalid norm value %s; should be "backward", "ortho", or "forward".' % norm)
    return norm