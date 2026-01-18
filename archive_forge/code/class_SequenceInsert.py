from typing import Any, List, Optional, Union
import numpy as np
from onnx.reference.op_run import OpRun
class SequenceInsert(OpRun):

    def _run(self, S, T, ind=None):
        if ind is None:
            res = sequence_insert_reference_implementation(S, T)
        elif isinstance(ind, int):
            res = sequence_insert_reference_implementation(S, T, [ind])
        elif len(ind.shape) > 0:
            res = sequence_insert_reference_implementation(S, T, ind)
        elif len(ind.shape) == 0:
            res = sequence_insert_reference_implementation(S, T, [int(ind)])
        else:
            res = sequence_insert_reference_implementation(S, T)
        return (res,)