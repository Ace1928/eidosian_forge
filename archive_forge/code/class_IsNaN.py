import numpy as np
from onnx.reference.ops._op import OpRunUnary
class IsNaN(OpRunUnary):

    def _run(self, data):
        return (np.isnan(data),)