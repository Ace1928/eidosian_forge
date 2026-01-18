import numpy as np
from onnx.reference.op_run import OpRun
class GatherElements(OpRun):

    def _run(self, data, indices, axis=None):
        if indices.size == 0:
            return (np.empty((0,), dtype=data.dtype),)
        try:
            return (gather_numpy(data, axis, indices),)
        except TypeError:
            return (gather_numpy(data, axis, indices.astype(int)),)