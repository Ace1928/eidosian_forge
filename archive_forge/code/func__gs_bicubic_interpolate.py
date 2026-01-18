import numbers
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_resize import _get_all_coords
def _gs_bicubic_interpolate(self, p, x, y):
    v = np.empty((4,), dtype=p.dtype)
    coeffs = np.empty((4,), dtype=p.dtype)
    self._gs_get_cubic_coeffs(x, coeffs)
    for i in range(4):
        v[i] = coeffs @ p[i, :]
    self._gs_get_cubic_coeffs(y, coeffs)
    return coeffs @ v