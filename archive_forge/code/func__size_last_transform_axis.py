import functools
import math
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._cache import get_plan_cache
def _size_last_transform_axis(shape, s, axes):
    if s is not None:
        if s[-1] is not None:
            return s[-1]
    elif axes is not None:
        return shape[axes[-1]]
    return shape[-1]