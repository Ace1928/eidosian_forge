import numpy as np
from onnx.reference.ops._op import OpRunReduceNumpy
class ReduceSum_1(OpRunReduceNumpy):

    def _run(self, x, axes=None, keepdims=None):
        axes = tuple(axes) if axes is not None else None
        res = np.sum(x, axis=axes, keepdims=keepdims, dtype=x.dtype)
        if keepdims == 0 and (not isinstance(res, np.ndarray)):
            res = np.array(res)
        return (res,)