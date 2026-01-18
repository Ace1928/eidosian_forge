import numbers
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_resize import _get_all_coords
def _gs_denormalize(self, n, length: int, align_corners: bool):
    if align_corners:
        x = (n + 1) / 2.0 * (length - 1)
    else:
        x = ((n + 1) * length - 1) / 2.0
    return x