import functools
import math
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._cache import get_plan_cache
def _get_cufft_plan_nd(shape, fft_type, axes=None, order='C', out_size=None, to_cache=True):
    """Generate a CUDA FFT plan for transforming up to three axes.

    Args:
        shape (tuple of int): The shape of the array to transform
        fft_type (int): The FFT type to perform. Supported values are:
            `cufft.CUFFT_C2C`, `cufft.CUFFT_C2R`, `cufft.CUFFT_R2C`,
            `cufft.CUFFT_Z2Z`, `cufft.CUFFT_Z2D`, and `cufft.CUFFT_D2Z`.
        axes (None or int or tuple of int):  The axes of the array to
            transform. Currently, these must be a set of up to three adjacent
            axes and must include either the first or the last axis of the
            array.  If `None`, it is assumed that all axes are transformed.
        order ({'C', 'F'}): Specify whether the data to be transformed has C or
            Fortran ordered data layout.
        out_size (int): The output length along the last axis for R2C/C2R FFTs.
            For C2C FFT, this is ignored (and set to `None`).
        to_cache (bool): Whether to cache the generated plan. Default is
            ``True``.

    Returns:
        plan (cufft.PlanNd): A cuFFT Plan for the chosen `fft_type`.
    """
    ndim = len(shape)
    if fft_type in (cufft.CUFFT_C2C, cufft.CUFFT_Z2Z):
        value_type = 'C2C'
    elif fft_type in (cufft.CUFFT_C2R, cufft.CUFFT_Z2D):
        value_type = 'C2R'
    else:
        value_type = 'R2C'
    if axes is None:
        fft_axes = tuple(range(ndim))
    else:
        _, fft_axes = _prep_fftn_axes(ndim, s=None, axes=axes, value_type=value_type)
    if not _nd_plan_is_possible(fft_axes, ndim):
        raise ValueError('An n-dimensional cuFFT plan could not be created. The axes must be contiguous and non-repeating. Between one and three axes can be transformed and either the first or last axis must be included in axes.')
    if order not in ['C', 'F']:
        raise ValueError("order must be 'C' or 'F'")
    '\n    For full details on idist, istride, iembed, etc. see:\n    http://docs.nvidia.com/cuda/cufft/index.html#advanced-data-layout\n\n    in 1D:\n    input[b * idist + x * istride]\n    output[b * odist + x * ostride]\n\n    in 2D:\n    input[b * idist + (x * inembed[1] + y) * istride]\n    output[b * odist + (x * onembed[1] + y) * ostride]\n\n    in 3D:\n    input[b * idist + ((x * inembed[1] + y) * inembed[2] + z) * istride]\n    output[b * odist + ((x * onembed[1] + y) * onembed[2] + z) * ostride]\n    '
    in_dimensions = [shape[d] for d in fft_axes]
    if order == 'F':
        in_dimensions = in_dimensions[::-1]
    in_dimensions = tuple(in_dimensions)
    if fft_type in (cufft.CUFFT_C2C, cufft.CUFFT_Z2Z):
        out_dimensions = in_dimensions
        plan_dimensions = in_dimensions
    else:
        out_dimensions = list(in_dimensions)
        if out_size is not None:
            out_dimensions[-1] = out_size
        out_dimensions = tuple(out_dimensions)
        if fft_type in (cufft.CUFFT_R2C, cufft.CUFFT_D2Z):
            plan_dimensions = in_dimensions
        else:
            plan_dimensions = out_dimensions
    inembed = in_dimensions
    onembed = out_dimensions
    if fft_axes == tuple(range(ndim)):
        nbatch = 1
        idist = odist = 1
        istride = ostride = 1
    elif 0 not in fft_axes:
        min_axis_fft = _reduce(min, fft_axes)
        nbatch = _prod(shape[:min_axis_fft])
        if order == 'C':
            idist = _prod(in_dimensions)
            odist = _prod(out_dimensions)
            istride = 1
            ostride = 1
        elif order == 'F':
            idist = 1
            odist = 1
            istride = nbatch
            ostride = nbatch
    elif ndim - 1 not in fft_axes:
        num_axes_batch = ndim - len(fft_axes)
        nbatch = _prod(shape[-num_axes_batch:])
        if order == 'C':
            idist = 1
            odist = 1
            istride = nbatch
            ostride = nbatch
        elif order == 'F':
            idist = _prod(in_dimensions)
            odist = _prod(out_dimensions)
            istride = 1
            ostride = 1
    else:
        raise ValueError('General subsets of FFT axes not currently supported for GPU case (Can only batch FFT over the first or last spatial axes).')
    for n in plan_dimensions:
        if n < 1:
            raise ValueError('Invalid number of FFT data points specified.')
    keys = (plan_dimensions, inembed, istride, idist, onembed, ostride, odist, fft_type, nbatch, order, fft_axes[-1], out_size)
    mgr = config.get_current_callback_manager()
    if mgr is not None:
        load_aux = mgr.cb_load_aux_arr
        store_aux = mgr.cb_store_aux_arr
        keys += (mgr.cb_load, mgr.cb_store, 0 if load_aux is None else load_aux.data.ptr, 0 if store_aux is None else store_aux.data.ptr)
    cache = get_plan_cache()
    cached_plan = cache.get(keys)
    if cached_plan is not None:
        plan = cached_plan
    elif mgr is None:
        plan = cufft.PlanNd(*keys)
        if to_cache:
            cache[keys] = plan
    else:
        plan = mgr.create_plan(('PlanNd', keys[:-4]))
        mgr.set_callbacks(plan)
        if to_cache:
            cache[keys] = plan
    return plan