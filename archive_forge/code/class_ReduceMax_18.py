import numpy as np
from onnx.reference.ops._op import OpRunReduceNumpy
class ReduceMax_18(OpRunReduceNumpy):

    def _run(self, data, axes=None, keepdims: int=1, noop_with_empty_axes: int=0):
        if self.is_axes_empty(axes) and noop_with_empty_axes != 0:
            return (data,)
        axes = self.handle_axes(axes)
        keepdims = keepdims != 0
        if data.size == 0:
            minvalue = np.iinfo(data.dtype).min if np.issubdtype(data.dtype, np.integer) else -np.inf
            return self.reduce_constant(data, minvalue, axes, keepdims)
        res = np.maximum.reduce(data, axis=axes, keepdims=keepdims)
        if keepdims == 0 and (not isinstance(res, np.ndarray)):
            res = np.array(res)
        return (res,)