import math
import os
import cupy
import numpy as np
from ._util import _get_inttype
from ._pba_2d import (_check_distances, _check_indices,
@cupy.memoize(for_each_device=True)
def _get_decode_as_distance_kernel(size_max, large_dist=False, sampling=None):
    """Fused decode3d and distance computation.

    This kernel is for use when `return_distances=True`, but
    `return_indices=False`. It replaces the separate calls to
    `_get_decode3d_kernel` and `_get_distance_kernel`, avoiding the overhead of
    generating full arrays containing the coordinates since the coordinate
    arrays are not going to be returned.
    """
    if sampling is None:
        dist_int_type = 'ptrdiff_t' if large_dist else 'int'
    int_type = 'int'
    code = _get_decode3d_code(size_max, int_type=int_type)
    code += _generate_shape(ndim=3, int_type=int_type, var_name='dist', raw_var=True)
    code += _generate_indices_ops(ndim=3, int_type=int_type)
    if sampling is None:
        code += _generate_distance_computation(int_type, dist_int_type)
        in_params = 'E encoded'
    else:
        code += _generate_aniso_distance_computation()
        in_params = 'E encoded, raw F sampling'
    return cupy.ElementwiseKernel(in_params=in_params, out_params='raw F dist', operation=code, options=('--std=c++11',))