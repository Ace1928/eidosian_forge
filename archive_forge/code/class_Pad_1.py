import numpy as np
from onnx.reference.op_run import OpRun
class Pad_1(OpRun):

    def _run(self, data, paddings=None, mode=None, value=None):
        if value is None:
            value = 0
        return (_pad_impl(data, paddings, mode=mode, constant_values=value),)