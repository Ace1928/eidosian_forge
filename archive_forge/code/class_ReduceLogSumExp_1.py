import numpy as np
from onnx.reference.ops._op import OpRunReduceNumpy
class ReduceLogSumExp_1(OpRunReduceNumpy):

    def _run(self, data, axes=None, keepdims=None):
        tax = tuple(axes) if axes is not None else None
        if data.size == 0:
            return self.reduce_constant(data, -np.inf, tax, keepdims)
        return compute_log_sum_exp(data, tax, keepdims)