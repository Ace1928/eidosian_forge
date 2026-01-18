import numbers
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_resize import _get_all_coords
def _pixel_at_ndarray(self, ndarray, x: List, border, padding_mode):
    num_dims = ndarray.ndim
    assert num_dims == len(x) == int(len(border) / 2)
    if num_dims == 1:
        return self._pixel_at_array(array=ndarray, i=x[0], border=border, padding_mode=padding_mode)
    i = x[0]
    d = ndarray.shape[0]
    if padding_mode == 'zeros':
        if i >= 0 and i < d:
            ndarray = ndarray[i]
        else:
            i = 0
            ndarray = np.zeros_like(ndarray[i])
    elif padding_mode == 'border':
        i = self._clamp(i, 0, d - 1)
        ndarray = ndarray[i]
    else:
        i = int(self._gs_reflect(i, border[0], border[num_dims]))
        ndarray = ndarray[i]
    return self._pixel_at_ndarray(ndarray=ndarray, x=x[1:], border=list(border[1:num_dims]) + list(border[1 + num_dims:2 * num_dims]), padding_mode=padding_mode)