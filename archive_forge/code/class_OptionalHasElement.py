import numpy as np
from onnx.reference.op_run import OpRun
class OptionalHasElement(OpRun):

    def _run(self, x=None):
        return (np.array(x is not None),)