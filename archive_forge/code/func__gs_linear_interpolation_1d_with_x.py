import numbers
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_resize import _get_all_coords
def _gs_linear_interpolation_1d_with_x(self, data, x, border, padding_mode):
    v = np.empty((2,), dtype=data.dtype)
    coeffs = np.empty((2,), dtype=data.dtype)
    x_0 = int(np.floor(x))
    x_1 = x_0 + 1
    self._gs_get_linear_coeffs(x - x_0, coeffs)
    v[0] = self._pixel_at_array(array=data, i=x_0, border=border, padding_mode=padding_mode)
    v[1] = self._pixel_at_array(array=data, i=x_1, border=border, padding_mode=padding_mode)
    return coeffs @ v