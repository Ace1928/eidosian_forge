from typing import Any, Dict
import numpy as np
from onnx.onnx_pb import NodeProto
from onnx.reference.op_run import OpRun, RuntimeTypeError
class OpRunBinaryNum(OpRunBinary):
    """Ancestor to all binary operators in this subfolder.

    Checks that input oud output types are the same.
    """

    def run(self, x, y):
        """Calls method ``OpRunBinary.run``, catches exceptions, displays a longer error message."""
        res = OpRunBinary.run(self, x, y)
        if res[0].dtype != x.dtype:
            raise RuntimeTypeError(f'Output type mismatch: {x.dtype} != {res[0].dtype} or {y.dtype} (operator {self.__class__.__name__!r}) type(x)={type(x)} type(y)={type(y)}')
        return self._check_and_fix_outputs(res)