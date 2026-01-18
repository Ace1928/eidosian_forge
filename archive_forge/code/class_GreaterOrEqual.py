import numpy as np
from onnx.reference.ops._op import OpRunBinaryComparison
class GreaterOrEqual(OpRunBinaryComparison):

    def _run(self, a, b):
        return (np.greater_equal(a, b),)