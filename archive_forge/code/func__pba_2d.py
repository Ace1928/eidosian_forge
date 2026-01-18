import math
import numbers
import os
import cupy
from ._util import _get_inttype
def _pba_2d(arr, sampling=None, return_distances=True, return_indices=False, block_params=None, check_warp_size=False, *, float64_distances=False, distances=None, indices=None):
    indices_inplace = isinstance(indices, cupy.ndarray)
    dt_inplace = isinstance(distances, cupy.ndarray)
    _distance_tranform_arg_check(dt_inplace, indices_inplace, return_distances, return_indices)
    block_size = _get_block_size(check_warp_size)
    if block_params is None:
        padded_size = math.ceil(max(arr.shape) / block_size) * block_size
        m1 = padded_size // block_size
        m2 = max(1, min(padded_size // block_size, block_size))
        m2 = 2 ** math.floor(math.log2(m2))
        if padded_size % m2 != 0:
            raise RuntimeError('error in setting default m2')
        m3 = min(min(m1, m2), 2)
    else:
        if any((p < 1 for p in block_params)):
            raise ValueError('(m1, m2, m3) in blockparams must be >= 1')
        m1, m2, m3 = block_params
        if math.log2(m2) % 1 > 1e-05:
            raise ValueError('m2 must be a power of 2')
        multiple = lcm(block_size, m1, m2, m3)
        padded_size = math.ceil(max(arr.shape) / multiple) * multiple
    if m1 > padded_size // block_size:
        raise ValueError(f'm1 too large. must be <= padded arr.shape[0] // {block_size}')
    if m2 > padded_size // block_size:
        raise ValueError(f'm2 too large. must be <= padded arr.shape[1] // {block_size}')
    if m3 > padded_size // block_size:
        raise ValueError(f'm3 too large. must be <= padded arr.shape[1] // {block_size}')
    for m in (m1, m2, m3):
        if padded_size % m != 0:
            raise ValueError(f'Largest dimension of image ({padded_size}) must be evenly disivible by each element of block_params: {(m1, m2, m3)}.')
    shape_max = max(arr.shape)
    if shape_max <= 32768:
        int_dtype = cupy.int16
        pixel_int2_type = 'short2'
    else:
        if shape_max > 1 << 24:
            raise ValueError(f'maximum axis size of {1 << 24} exceeded, for image with shape {arr.shape}')
        int_dtype = cupy.int32
        pixel_int2_type = 'int2'
    marker = _init_marker(int_dtype)
    orig_sy, orig_sx = arr.shape
    padding_width = _determine_padding(arr.shape, padded_size, block_size)
    if padding_width is not None:
        arr = cupy.pad(arr, padding_width, mode='constant', constant_values=1)
    size = arr.shape[0]
    input_arr = _pack_int2(arr, marker=marker, int_dtype=int_dtype)
    output = cupy.zeros_like(input_arr)
    int2_dtype = cupy.dtype({'names': ['x', 'y'], 'formats': [int_dtype] * 2})
    margin = cupy.empty((2 * m1 * size,), dtype=int2_dtype)
    pba2d = cupy.RawModule(code=get_pba2d_src(block_size_2d=block_size, marker=marker, pixel_int2_t=pixel_int2_type))
    kernelFloodDown = pba2d.get_function('kernelFloodDown')
    kernelFloodUp = pba2d.get_function('kernelFloodUp')
    kernelPropagateInterband = pba2d.get_function('kernelPropagateInterband')
    kernelUpdateVertical = pba2d.get_function('kernelUpdateVertical')
    kernelCreateForwardPointers = pba2d.get_function('kernelCreateForwardPointers')
    kernelDoubleToSingleList = pba2d.get_function('kernelDoubleToSingleList')
    if sampling is None:
        kernelProximatePoints = pba2d.get_function('kernelProximatePoints')
        kernelMergeBands = pba2d.get_function('kernelMergeBands')
        kernelColor = pba2d.get_function('kernelColor')
    else:
        kernelProximatePoints = pba2d.get_function('kernelProximatePointsWithSpacing')
        kernelMergeBands = pba2d.get_function('kernelMergeBandsWithSpacing')
        kernelColor = pba2d.get_function('kernelColorWithSpacing')
    block = (block_size, 1, 1)
    grid = (math.ceil(size / block[0]), m1, 1)
    bandSize1 = size // m1
    kernelFloodDown(grid, block, (input_arr, input_arr, size, bandSize1))
    kernelFloodUp(grid, block, (input_arr, input_arr, size, bandSize1))
    kernelPropagateInterband(grid, block, (input_arr, margin, size, bandSize1))
    kernelUpdateVertical(grid, block, (input_arr, margin, output, size, bandSize1))
    block = (block_size, 1, 1)
    grid = (math.ceil(size / block[0]), m2, 1)
    bandSize2 = size // m2
    if sampling is None:
        sampling_args = ()
    else:
        sampling = tuple(map(float, sampling))
        sampling_args = (sampling[0], sampling[1])
    kernelProximatePoints(grid, block, (output, input_arr, size, bandSize2) + sampling_args)
    kernelCreateForwardPointers(grid, block, (input_arr, input_arr, size, bandSize2))
    noBand = m2
    while noBand > 1:
        grid = (math.ceil(size / block[0]), noBand // 2)
        kernelMergeBands(grid, block, (output, input_arr, input_arr, size, size // noBand) + sampling_args)
        noBand //= 2
    grid = (math.ceil(size / block[0]), size)
    kernelDoubleToSingleList(grid, block, (output, input_arr, input_arr, size))
    block = (block_size, m3, 1)
    grid = (math.ceil(size / block[0]), 1, 1)
    kernelColor(grid, block, (input_arr, output, size) + sampling_args)
    output = _unpack_int2(output, make_copy=False, int_dtype=int_dtype)
    x = output[:orig_sy, :orig_sx, 0]
    y = output[:orig_sy, :orig_sx, 1]
    vals = ()
    if return_distances:
        dtype_out = cupy.float64 if float64_distances else cupy.float32
        if dt_inplace:
            _check_distances(distances, y.shape, dtype_out)
        else:
            distances = cupy.zeros(y.shape, dtype=dtype_out)
        max_possible_dist = sum(((s - 1) ** 2 for s in y.shape))
        dist_int_type = 'int' if max_possible_dist < 2 ** 31 else 'ptrdiff_t'
        if sampling is None:
            distance_kernel = _get_distance_kernel(int_type=_get_inttype(distances), dist_int_type=dist_int_type)
            distance_kernel(y, x, distances, size=distances.size)
        else:
            distance_kernel = _get_aniso_distance_kernel(int_type=_get_inttype(distances))
            sampling = cupy.asarray(sampling, dtype=dtype_out)
            distance_kernel(y, x, sampling, distances, size=distances.size)
        vals = vals + (distances,)
    if return_indices:
        if indices_inplace:
            _check_indices(indices, (arr.ndim,) + arr.shape, x.dtype.itemsize)
            indices[0, ...] = y
            indices[1, ...] = x
        else:
            indices = cupy.stack((y, x), axis=0)
        vals = vals + (indices,)
    return vals