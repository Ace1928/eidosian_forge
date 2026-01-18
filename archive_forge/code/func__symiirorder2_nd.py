import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupy._core.internal import _normalize_axis_index
from cupyx.scipy.signal._signaltools import lfilter
from cupyx.scipy.signal._arraytools import (
from cupyx.scipy.signal._iir_utils import collapse_2d, apply_iir_sos
def _symiirorder2_nd(input, r, omega, precision=-1.0, axis=-1):
    if r >= 1.0:
        raise ValueError('r must be less than 1.0')
    if precision <= 0.0 or precision > 1.0:
        if input.dtype is cupy.dtype(cupy.float64):
            precision = 1e-11
        elif input.dtype is cupy.dtype(cupy.float32):
            precision = 1e-06
        else:
            precision = 10 ** (-cupy.finfo(input.dtype).iexp)
    axis = _normalize_axis_index(axis, input.ndim)
    input_shape = input.shape
    input_ndim = input.ndim
    if input.ndim > 1:
        input, input_shape = collapse_2d(input, axis)
    block_sz = 128
    rsq = r * r
    a2 = 2 * r * cupy.cos(omega)
    a3 = -rsq
    cs = cupy.atleast_1d(1 - 2 * r * cupy.cos(omega) + rsq)
    omega = cupy.asarray(omega, cs.dtype)
    r = cupy.asarray(r, cs.dtype)
    rsq = cupy.asarray(rsq, cs.dtype)
    precision *= precision
    compute_symiirorder2_fwd_sc = _get_module_func(SYMIIR2_MODULE, 'compute_symiirorder2_fwd_sc', cs)
    diff = cupy.empty((block_sz + 1,), dtype=cs.dtype)
    all_valid = cupy.empty((block_sz + 1,), dtype=cupy.bool_)
    starting_diff = cupy.arange(2, dtype=input.dtype)
    starting_diff = _compute_symiirorder2_fwd_hc(starting_diff, cs, r, omega)
    y0 = cupy.nan
    y1 = cupy.nan
    for i in range(0, input.shape[-1] + 2, block_sz):
        compute_symiirorder2_fwd_sc((1,), (block_sz + 1,), (input.shape[-1] + 2, i, cs, r, omega, precision, all_valid, diff))
        input_slice = axis_slice(input, i, i + block_sz)
        diff_y0 = diff[:-1][:input_slice.shape[-1]]
        diff_y1 = diff[1:][:input_slice.shape[-1]]
        if cupy.isnan(y0):
            cum_poly_y0 = cupy.cumsum(diff_y0 * input_slice, axis=-1) + starting_diff[0] * axis_slice(input, 0, 1)
            y0 = _find_initial_cond(all_valid[:-1][:input_slice.shape[-1]], cum_poly_y0, input.shape[-1], i)
        if cupy.isnan(y1):
            cum_poly_y1 = cupy.cumsum(diff_y1 * input_slice, axis=-1) + starting_diff[0] * axis_slice(input, 1, 2) + starting_diff[1] * axis_slice(input, 0, 1)
            y1 = _find_initial_cond(all_valid[1:][:input_slice.shape[-1]], cum_poly_y1, input.shape[-1], i)
        if not cupy.any(cupy.isnan(cupy.r_[y0, y1])):
            break
    if cupy.any(cupy.isnan(cupy.r_[y0, y1])):
        raise ValueError('Sum to find symmetric boundary conditions did not converge.')
    zi_shape = (1, 4)
    if input_ndim > 1:
        zi_shape = (1, input.shape[0], 4)
    sos = cupy.atleast_2d(cupy.r_[cs, 0, 0, 1, -a2, -a3])
    sos = sos.astype(input.dtype)
    all_zi = cupy.zeros(zi_shape, dtype=input.dtype)
    all_zi = axis_assign(all_zi, y0, 2, 3)
    all_zi = axis_assign(all_zi, y1, 3, 4)
    y_fwd, _ = apply_iir_sos(axis_slice(input, 2), sos, zi=all_zi, dtype=input.dtype)
    if input_ndim > 1:
        y_fwd = cupy.c_[y0, y1, y_fwd]
    else:
        y_fwd = cupy.r_[y0, y1, y_fwd]
    compute_symiirorder2_bwd_sc = _get_module_func(SYMIIR2_MODULE, 'compute_symiirorder2_bwd_sc', cs)
    diff = cupy.empty((block_sz,), dtype=cs.dtype)
    all_valid = cupy.empty((block_sz,), dtype=cupy.bool_)
    rev_input = axis_reverse(input)
    y0 = cupy.nan
    for i in range(0, input.shape[-1] + 1, block_sz):
        compute_symiirorder2_bwd_sc((1,), (block_sz,), (input.shape[-1] + 1, i, 0, 1, cs, cupy.asarray(rsq, cs.dtype), cupy.asarray(omega, cs.dtype), precision, all_valid, diff))
        input_slice = axis_slice(rev_input, i, i + block_sz)
        cum_poly_y0 = cupy.cumsum(diff[:input_slice.shape[-1]] * input_slice, axis=-1)
        y0 = _find_initial_cond(all_valid[:input_slice.shape[-1]], cum_poly_y0, input.shape[-1], i)
        if not cupy.any(cupy.isnan(y0)):
            break
    if cupy.any(cupy.isnan(y0)):
        raise ValueError('Sum to find symmetric boundary conditions did not converge.')
    y1 = cupy.nan
    for i in range(0, input.shape[-1] + 1, block_sz):
        compute_symiirorder2_bwd_sc((1,), (block_sz,), (input.size + 1, i, -1, 2, cs, cupy.asarray(rsq, cs.dtype), cupy.asarray(omega, cs.dtype), precision, all_valid, diff))
        input_slice = axis_slice(rev_input, i, i + block_sz)
        cum_poly_y1 = cupy.cumsum(diff[:input_slice.shape[-1]] * input_slice, axis=-1)
        y1 = _find_initial_cond(all_valid[:input_slice.size], cum_poly_y1, input.size, i)
        if not cupy.any(cupy.isnan(y1)):
            break
    if cupy.any(cupy.isnan(y1)):
        raise ValueError('Sum to find symmetric boundary conditions did not converge.')
    all_zi = axis_assign(all_zi, y0, 2, 3)
    all_zi = axis_assign(all_zi, y1, 3, 4)
    out, _ = apply_iir_sos(axis_slice(y_fwd, -3, step=-1), sos, zi=all_zi)
    if input_ndim > 1:
        out = cupy.c_[axis_reverse(out), y1, y0]
    else:
        out = cupy.r_[axis_reverse(out), y1, y0]
    if input_ndim > 1:
        out = out.reshape(input_shape)
        out = cupy.moveaxis(out, -1, axis)
        if not out.flags.c_contiguous:
            out = out.copy()
    return out