import numpy as np
from onnx.reference.ops._op import OpRunReduceNumpy
class ReduceLogSum_18(OpRunReduceNumpy):

    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        if self.is_axes_empty(axes) and noop_with_empty_axes:
            return (data,)
        axes = self.handle_axes(axes)
        keepdims = keepdims != 0
        if data.size == 0:
            return self.reduce_constant(data, -np.inf, axes, keepdims)
        res = np.sum(data, axis=axes, keepdims=keepdims)
        if len(res.shape) > 0:
            return (np.log(res, out=res),)
        return (np.log(res),)