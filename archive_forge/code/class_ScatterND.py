import numpy as np
from onnx.reference.op_run import OpRun
class ScatterND(OpRun):

    def _run(self, data, indices, updates, reduction=None):
        y = _scatter_nd_impl(data, indices, updates, reduction=reduction)
        return (y,)