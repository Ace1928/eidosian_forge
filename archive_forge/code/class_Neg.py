import numpy as np
from onnx.reference.ops._op import OpRunUnaryNum
class Neg(OpRunUnaryNum):

    def _run(self, x):
        return (np.negative(x),)