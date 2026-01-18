import numbers
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_resize import _get_all_coords
def _prepare_border(self, dims, align_corners: bool):
    num_dims = len(dims)
    borders = np.zeros(num_dims * 2)
    for i in range(num_dims):
        borders[i] = -0.5
        borders[i + num_dims] = dims[i] - 0.5
        if align_corners:
            borders[i] = 0.0
            borders[i + num_dims] = dims[i] - 1.0
    return borders