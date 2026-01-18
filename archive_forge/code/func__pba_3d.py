import math
import os
import cupy
import numpy as np
from ._util import _get_inttype
from ._pba_2d import (_check_distances, _check_indices,
def _pba_3d(arr, sampling=None, return_distances=True, return_indices=False, block_params=None, check_warp_size=False, *, float64_distances=False, distances=None, indices=None):
    indices_inplace = isinstance(indices, cupy.ndarray)
    dt_inplace = isinstance(distances, cupy.ndarray)
    _distance_tranform_arg_check(dt_inplace, indices_inplace, return_distances, return_indices)
    if arr.ndim != 3:
        raise ValueError(f'expected a 3D array, got {arr.ndim}D')
    if block_params is None:
        m1 = 1
        m2 = 1
        m3 = 2
    else:
        m1, m2, m3 = block_params
    s_min = min(arr.shape)
    if s_min <= 4:
        blockx = 4
    elif s_min <= 8:
        blockx = 8
    elif s_min <= 16:
        blockx = 16
    else:
        blockx = 32
    blocky = 4
    block_size = _get_block_size(check_warp_size)
    orig_sz, orig_sy, orig_sx = arr.shape
    padding_width = _determine_padding(arr.shape, block_size, m1, m2, m3, blockx, blocky)
    if padding_width is not None:
        arr = cupy.pad(arr, padding_width, mode='constant', constant_values=1)
    size = arr.shape[0]
    size_max = max(arr.shape)
    input_arr = encode3d(arr, size_max=size_max)
    buffer_idx = 0
    output = cupy.zeros_like(input_arr)
    pba_images = [input_arr, output]
    block = (blockx, blocky, 1)
    grid = (size // block[0], size // block[1], 1)
    pba3d = cupy.RawModule(code=get_pba3d_src(block_size_3d=block_size, size_max=size_max))
    kernelFloodZ = pba3d.get_function('kernelFloodZ')
    if sampling is None:
        kernelMaurerAxis = pba3d.get_function('kernelMaurerAxis')
        kernelColorAxis = pba3d.get_function('kernelColorAxis')
        sampling_args = ()
    else:
        kernelMaurerAxis = pba3d.get_function('kernelMaurerAxisWithSpacing')
        kernelColorAxis = pba3d.get_function('kernelColorAxisWithSpacing')
        sampling = tuple(map(float, sampling))
        sampling_args = (sampling[2], sampling[1], sampling[0])
    kernelFloodZ(grid, block, (pba_images[buffer_idx], pba_images[1 - buffer_idx], size))
    buffer_idx = 1 - buffer_idx
    block = (blockx, blocky, 1)
    grid = (size // block[0], size // block[1], 1)
    kernelMaurerAxis(grid, block, (pba_images[buffer_idx], pba_images[1 - buffer_idx], size) + sampling_args)
    block = (block_size, m3, 1)
    grid = (size // block[0], size, 1)
    kernelColorAxis(grid, block, (pba_images[1 - buffer_idx], pba_images[buffer_idx], size) + sampling_args)
    if sampling is not None:
        sampling_args = (sampling[1], sampling[2], sampling[0])
    block = (blockx, blocky, 1)
    grid = (size // block[0], size // block[1], 1)
    kernelMaurerAxis(grid, block, (pba_images[buffer_idx], pba_images[1 - buffer_idx], size) + sampling_args)
    block = (block_size, m3, 1)
    grid = (size // block[0], size, 1)
    kernelColorAxis(grid, block, (pba_images[1 - buffer_idx], pba_images[buffer_idx], size) + sampling_args)
    output = pba_images[buffer_idx]
    if return_distances:
        out_shape = (orig_sz, orig_sy, orig_sx)
        dtype_out = cupy.float64 if float64_distances else cupy.float32
        if dt_inplace:
            _check_distances(distances, out_shape, dtype_out)
        else:
            distances = cupy.zeros(out_shape, dtype=dtype_out)
        max_possible_dist = sum(((s - 1) ** 2 for s in out_shape))
        large_dist = max_possible_dist >= 2 ** 31
        if not return_indices:
            kern = _get_decode_as_distance_kernel(size_max=size_max, large_dist=large_dist, sampling=sampling)
            if sampling is None:
                kern(output[:orig_sz, :orig_sy, :orig_sx], distances)
            else:
                sampling = cupy.asarray(sampling, dtype=distances.dtype)
                kern(output[:orig_sz, :orig_sy, :orig_sx], sampling, distances)
            return (distances,)
    if return_indices:
        x, y, z = decode3d(output[:orig_sz, :orig_sy, :orig_sx], size_max=size_max)
    vals = ()
    if return_distances:
        if sampling is None:
            kern = _get_distance_kernel(int_type=_get_inttype(distances), large_dist=large_dist)
            kern(z, y, x, distances)
        else:
            kern = _get_aniso_distance_kernel(int_type=_get_inttype(distances))
            sampling = cupy.asarray(sampling, dtype=distances.dtype)
            kern(z, y, x, sampling, distances)
        vals = vals + (distances,)
    if return_indices:
        if indices_inplace:
            _check_indices(indices, (arr.ndim,) + arr.shape, x.dtype.itemsize)
            indices[0, ...] = z
            indices[1, ...] = y
            indices[2, ...] = x
        else:
            indices = cupy.stack((z, y, x), axis=0)
        vals = vals + (indices,)
    return vals