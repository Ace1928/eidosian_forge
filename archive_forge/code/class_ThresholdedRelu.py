import numpy as np
from onnx.reference.ops._op import OpRunUnaryNum
class ThresholdedRelu(OpRunUnaryNum):

    def _run(self, x, alpha=None):
        alpha = alpha or self.alpha
        return (np.where(x > alpha, x, 0).astype(x.dtype),)