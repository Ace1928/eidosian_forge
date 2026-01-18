from typing import Optional
import numpy as np
from onnx.reference.ops._op import OpRun
class SliceCommon(OpRun):

    def _run(self, data, starts, ends, axes=None, steps=None):
        res = _slice(data, starts, ends, axes, steps)
        return (res,)