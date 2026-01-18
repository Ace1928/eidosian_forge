import numpy as np
from onnx.reference.op_run import OpRun
class NonZero(OpRun):

    def _run(self, x):
        res = np.vstack(np.nonzero(x)).astype(np.int64)
        return (res,)