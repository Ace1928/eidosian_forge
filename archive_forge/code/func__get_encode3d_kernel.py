import math
import os
import cupy
import numpy as np
from ._util import _get_inttype
from ._pba_2d import (_check_distances, _check_indices,
@cupy.memoize(for_each_device=True)
def _get_encode3d_kernel(size_max, marker=-2147483648):
    """Pack array coordinates into a single integer."""
    if size_max > 1024:
        int_type = 'ptrdiff_t'
    else:
        int_type = 'int'
    if size_max > 1024:
        value = '(((x) << 40) | ((y) << 20) | (z))'
    else:
        value = '(((x) << 20) | ((y) << 10) | (z))'
    code = f'\n    if (arr[i]) {{\n        out[i] = {marker};\n    }} else {{\n        {int_type} shape_2 = arr.shape()[2];\n        {int_type} shape_1 = arr.shape()[1];\n        {int_type} _i = i;\n        {int_type} x = _i % shape_2;\n        _i /= shape_2;\n        {int_type} y = _i % shape_1;\n        _i /= shape_1;\n        {int_type} z = _i;\n        out[i] = {value};\n    }}\n    '
    return cupy.ElementwiseKernel(in_params='raw B arr', out_params='raw I out', operation=code, options=('--std=c++11',))