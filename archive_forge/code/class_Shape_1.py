from typing import Optional, Tuple
import numpy as np
from onnx.reference.op_run import OpRun
class Shape_1(OpRun):

    def _run(self, data):
        return (np.array(data.shape, dtype=np.int64),)