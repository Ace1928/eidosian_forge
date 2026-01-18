import numpy as np
from onnx.reference.op_run import OpRun
class GlobalMaxPool(OpRun):

    def _run(self, x):
        res = _global_max_pool(x)
        return (res,)