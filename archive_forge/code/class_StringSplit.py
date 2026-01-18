from typing import Union
import numpy as np
from onnx.reference.op_run import OpRun
class StringSplit(OpRun):

    def _run(self, x, delimiter=None, maxsplit=None):
        if delimiter == '':
            delimiter = None
        if x.dtype.kind not in _acceptable_str_dtypes:
            raise TypeError(f'Inputs must be string tensors, received dtype {x.dtype}')
        return split_with_padding(x, delimiter, maxsplit)