import numpy as np
from onnx.reference.ops._op import OpRunUnaryNum
class Acosh(OpRunUnaryNum):

    def _run(self, x):
        return (np.arccosh(x),)