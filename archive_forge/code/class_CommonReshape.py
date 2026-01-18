import numpy as np
from onnx.reference.op_run import OpRun
class CommonReshape(OpRun):

    def _run(self, data, shape):
        return (reshape_reference_implementation(data, shape, 0),)