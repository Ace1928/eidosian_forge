import numpy as np
from onnx.reference.ops._op import OpRunReduceNumpy
class ReduceMax_1(OpRunReduceNumpy):

    def _run(self, data, axes=None, keepdims=None):
        axes = tuple(axes) if axes is not None else None
        if data.size == 0:
            minvalue = np.iinfo(data.dtype).min if np.issubdtype(data.dtype, np.integer) else -np.inf
            return self.reduce_constant(data, minvalue, axes, keepdims)
        res = np.maximum.reduce(data, axis=axes, keepdims=keepdims == 1)
        if keepdims == 0 and (not isinstance(res, np.ndarray)):
            res = np.array(res)
        return (res,)