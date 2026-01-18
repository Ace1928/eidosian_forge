import numpy as np
from onnx.reference.op_run import OpRun
class Pad_11(OpRun):

    def _run(self, data, pads, constant_value=None, mode=None):
        if constant_value is None:
            constant_value = 0
        return (_pad_impl(data, pads, mode=mode, constant_values=constant_value, axes=None),)