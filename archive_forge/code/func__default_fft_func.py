import functools
import math
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._cache import get_plan_cache
def _default_fft_func(a, s=None, axes=None, plan=None, value_type='C2C'):
    curr_plan = cufft.get_current_plan()
    if curr_plan is not None:
        if plan is None:
            plan = curr_plan
        else:
            raise RuntimeError('Use the cuFFT plan either as a context manager or as an argument.')
    if isinstance(plan, cufft.PlanNd):
        return _fftn
    elif isinstance(plan, cufft.Plan1d) or a.ndim == 1 or (not config.enable_nd_planning):
        return _fft
    if a.flags.f_contiguous and value_type != 'C2C':
        return _fft
    _, axes_sorted = _prep_fftn_axes(a.ndim, s, axes, value_type)
    if len(axes_sorted) > 1 and _nd_plan_is_possible(axes_sorted, a.ndim):
        if cupy.cuda.runtime.is_hip:
            if 0 == axes_sorted[0] and len(axes_sorted) != a.ndim and a.flags.c_contiguous:
                return _fft
            if value_type == 'C2R':
                return _fft
        return _fftn
    return _fft