import functools
import math
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._cache import get_plan_cache
def _fftn(a, s, axes, norm, direction, value_type='C2C', order='A', plan=None, overwrite_x=False, out=None):
    if not isinstance(a, cupy.ndarray):
        raise TypeError('The input array a must be a cupy.ndarray')
    if norm is None:
        norm = 'backward'
    if norm not in ('backward', 'ortho', 'forward'):
        raise ValueError('Invalid norm value %s, should be "backward", "ortho", or "forward".' % norm)
    axes, axes_sorted = _prep_fftn_axes(a.ndim, s, axes, value_type)
    if not axes_sorted:
        if value_type == 'C2C':
            return a
        else:
            raise IndexError('list index out of range')
    a = _convert_dtype(a, value_type)
    if order == 'A':
        if a.flags.f_contiguous:
            order = 'F'
        elif a.flags.c_contiguous:
            order = 'C'
        else:
            a = cupy.ascontiguousarray(a)
            order = 'C'
    elif order not in ['C', 'F']:
        raise ValueError('Unsupported order: {}'.format(order))
    a = _cook_shape(a, s, axes, value_type, order=order)
    for n in a.shape:
        if n < 1:
            raise ValueError('Invalid number of FFT data points (%d) specified.' % n)
    if order == 'C' and (not a.flags.c_contiguous):
        a = cupy.ascontiguousarray(a)
    elif order == 'F' and (not a.flags.f_contiguous):
        a = cupy.asfortranarray(a)
    out_size = _get_fftn_out_size(a.shape, s, axes_sorted[-1], value_type)
    a = _exec_fftn(a, direction, value_type, norm=norm, axes=axes_sorted, overwrite_x=overwrite_x, plan=plan, out=out, out_size=out_size)
    return a