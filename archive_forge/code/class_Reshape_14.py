import numpy as np
from onnx.reference.op_run import OpRun
class Reshape_14(CommonReshape):

    def _run(self, data, shape, allowzero=None):
        if allowzero is None:
            allowzero = getattr(self, 'allowzero', 0) == 1
        else:
            allowzero = allowzero == 1
        return (reshape_reference_implementation(data, shape, allowzero),)