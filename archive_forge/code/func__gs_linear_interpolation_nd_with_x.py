import numbers
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_resize import _get_all_coords
def _gs_linear_interpolation_nd_with_x(self, data, x, border, padding_mode):
    num_dims = data.ndim
    assert num_dims == len(x) == int(len(border) / 2)
    if num_dims == 1:
        return self._gs_linear_interpolation_1d_with_x(data=data, x=x[0], border=border, padding_mode=padding_mode)
    res1d = []
    for i in range(data.shape[0]):
        r = self._gs_linear_interpolation_nd_with_x(data=data[i], x=x[1:], border=list(border[1:num_dims]) + list(border[1 + num_dims:2 * num_dims]), padding_mode=padding_mode)
        res1d.append(r)
    res1d = np.array(res1d)
    return self._gs_linear_interpolation_1d_with_x(data=res1d, x=x[0], border=[border[0], border[num_dims]], padding_mode=padding_mode)