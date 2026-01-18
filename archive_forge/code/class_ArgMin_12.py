import numpy as np
from onnx.reference.op_run import OpRun
class ArgMin_12(_ArgMin):

    def _run(self, data, axis=None, keepdims=None, select_last_index=None):
        if select_last_index == 0:
            return _ArgMin._run(self, data, axis=axis, keepdims=keepdims)
        return (_argmin_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims),)