from typing import Dict, List
from unittest.mock import patch
import sympy
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, IRNode, Pointwise
from torch._inductor.utils import IndentedBuffer, sympy_str
def _op_constant(self, value, dtype):
    if str(dtype) in ('torch.float16', 'torch.float32'):
        return '{ static_cast<ElementAcc>(' + str(value) + ') }'
    else:
        raise CUTLASSEVTOpNotImplementedError(f'Unsupported dtype for constant: {dtype}')