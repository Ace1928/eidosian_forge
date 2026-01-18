import numpy as np
from onnx.reference.op_run import OpRun
class Celu(OpRun):

    def _run(self, x, alpha=None):
        return (_vcelu1(x, alpha).astype(x.dtype),)