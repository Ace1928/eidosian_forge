from typing import Any, Dict
import numpy as np
from onnx.onnx_pb import NodeProto
from onnx.reference.op_run import OpRun, RuntimeTypeError
class OpRunBinary(OpRun):
    """Ancestor to all binary operators in this subfolder.

    Checks that input and output types are the same.
    """

    def run(self, x, y):
        """Calls method ``_run``, catches exceptions, displays a longer error message.

        Supports only binary operators.
        """
        self._log('-- begin %s.run(2 inputs)', self.__class__.__name__)
        if x is None or y is None:
            raise RuntimeError(f'x and y have different dtype: {type(x)} != {type(y)} ({type(self)})')
        if x.dtype != y.dtype:
            raise RuntimeTypeError(f"Input type mismatch: {x.dtype} != {y.dtype} (operator '{self.__class__.__name__!r}', shapes {x.shape}, {y.shape}).")
        try:
            res = self._run(x, y)
        except (TypeError, ValueError) as e:
            raise TypeError(f'Issues with types {', '.join((str(type(_)) for _ in [x, y]))} (binary operator {self.__class__.__name__!r}).') from e
        self._log('-- done %s.run -> %d outputs', self.__class__.__name__, len(res))
        return self._check_and_fix_outputs(res)