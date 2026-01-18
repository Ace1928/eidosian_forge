import warnings
import cupy
from cupy import _core
from cupy._core import _routines_statistics as _statistics
from cupy._core import _fusion_thread_local
from cupy._logic import content
def _quantile_unchecked(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False):
    if q.ndim == 0:
        q = q[None]
        zerod = True
    else:
        zerod = False
    if q.ndim > 1:
        raise ValueError('Expected q to have a dimension of 1.\nActual: {0} != 1'.format(q.ndim))
    if keepdims:
        if axis is None:
            keepdim = (1,) * a.ndim
        else:
            keepdim = list(a.shape)
            for ax in axis:
                keepdim[ax % a.ndim] = 1
            keepdim = tuple(keepdim)
    if isinstance(axis, int):
        axis = (axis,)
    if axis is None:
        if overwrite_input:
            ap = a.ravel()
        else:
            ap = a.flatten()
        nkeep = 0
    else:
        axis = tuple((ax % a.ndim for ax in axis))
        keep = set(range(a.ndim)) - set(axis)
        nkeep = len(keep)
        for i, s in enumerate(sorted(keep)):
            a = a.swapaxes(i, s)
        if overwrite_input:
            ap = a.reshape(a.shape[:nkeep] + (-1,))
        else:
            ap = a.reshape(a.shape[:nkeep] + (-1,)).copy()
    axis = -1
    ap.sort(axis=axis)
    Nx = ap.shape[axis]
    indices = q * (Nx - 1.0)
    if method in ['inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf', 'hazen', 'weibull', 'median_unbiased', 'normal_unbiased']:
        raise ValueError(f"'{method}' method is not yet supported. Please use any other method.")
    elif method == 'lower':
        indices = cupy.floor(indices).astype(cupy.int32)
    elif method == 'higher':
        indices = cupy.ceil(indices).astype(cupy.int32)
    elif method == 'midpoint':
        indices = 0.5 * (cupy.floor(indices) + cupy.ceil(indices))
    elif method == 'nearest':
        raise ValueError("'nearest' method is not yet supported. Please use any other method.")
    elif method == 'linear':
        pass
    else:
        raise ValueError("Unexpected interpolation method.\nActual: '{0}' not in ('linear', 'lower', 'higher', 'midpoint')".format(method))
    if indices.dtype == cupy.int32:
        ret = cupy.rollaxis(ap, axis)
        ret = ret.take(indices, axis=0, out=out)
    else:
        if out is None:
            ret = cupy.empty(ap.shape[:-1] + q.shape, dtype=cupy.float64)
        else:
            ret = cupy.rollaxis(out, 0, out.ndim)
        cupy.ElementwiseKernel('S idx, raw T a, raw int32 offset, raw int32 size', 'U ret', '\n            ptrdiff_t idx_below = floor(idx);\n            U weight_above = idx - idx_below;\n\n            ptrdiff_t max_idx = size - 1;\n            ptrdiff_t offset_bottom = _ind.get()[0] * offset + idx_below;\n            ptrdiff_t offset_top = min(offset_bottom + 1, max_idx);\n\n            U diff = a[offset_top] - a[offset_bottom];\n\n            if (weight_above < 0.5) {\n                ret = a[offset_bottom] + diff * weight_above;\n            } else {\n                ret = a[offset_top] - diff * (1 - weight_above);\n            }\n            ', 'cupy_percentile_weightnening')(indices, ap, ap.shape[-1] if ap.ndim > 1 else 0, ap.size, ret)
        ret = cupy.rollaxis(ret, -1)
    if zerod:
        ret = ret.squeeze(0)
    if keepdims:
        if q.size > 1:
            keepdim = (-1,) + keepdim
        ret = ret.reshape(keepdim)
    return _core._internal_ascontiguousarray(ret)