import functools
import math
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._cache import get_plan_cache
def _nd_plan_is_possible(axes_sorted, ndim):
    return 0 < len(axes_sorted) <= 3 and (0 in axes_sorted or ndim - 1 in axes_sorted) and all((axes_sorted[n + 1] - axes_sorted[n] == 1 for n in range(len(axes_sorted) - 1)))