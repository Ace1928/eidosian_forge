import numbers
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_resize import _get_all_coords
def _gs_denormalize_coordinates(self, n, dims, align_corners: bool):
    x = np.zeros(len(n), dtype=np.float32)
    for i, (v, dim) in enumerate(zip(n, dims)):
        x[i] = self._gs_denormalize(n=v, length=dim, align_corners=align_corners)
    return x