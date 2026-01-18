from typing import Tuple
import numpy as np
from onnx.reference.op_run import OpRun
class GatherND(OpRun):

    def _run(self, data, indices, batch_dims=None):
        return _gather_nd_impl(data, indices, batch_dims)