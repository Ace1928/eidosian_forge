import numpy as np
from onnx.reference.op_run import OpRun
class _ArgMax(OpRun):

    def _run(self, data, axis=None, keepdims=None):
        return (_argmax(data, axis=axis, keepdims=keepdims),)