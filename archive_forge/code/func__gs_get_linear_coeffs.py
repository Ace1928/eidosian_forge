import numbers
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_resize import _get_all_coords
def _gs_get_linear_coeffs(self, x, coeffs):
    x = abs(x)
    coeffs[0] = 1 - x
    coeffs[1] = x