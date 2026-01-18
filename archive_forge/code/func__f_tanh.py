import numpy as np
from onnx.reference.op_run import OpRun
def _f_tanh(self, x):
    return np.tanh(x)