import numbers
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_resize import _get_all_coords
def _gs_get_cubic_coeffs(self, x, coeffs):
    """Calculate cubic convolution interpolation coefficients
        ROBERT G. KEYS https://ieeexplore.ieee.org/document/1163711
        Use float to avoid potential issues with integer.
        """
    cubic_alpha = -0.75
    x = abs(x)
    coeffs[0] = ((cubic_alpha * (x + 1) - 5 * cubic_alpha) * (x + 1) + 8 * cubic_alpha) * (x + 1) - 4 * cubic_alpha
    coeffs[1] = ((cubic_alpha + 2) * x - (cubic_alpha + 3)) * x * x + 1
    coeffs[2] = ((cubic_alpha + 2) * (1 - x) - (cubic_alpha + 3)) * (1 - x) * (1 - x) + 1
    coeffs[3] = ((cubic_alpha * (2 - x) - 5 * cubic_alpha) * (2 - x) + 8 * cubic_alpha) * (2 - x) - 4 * cubic_alpha