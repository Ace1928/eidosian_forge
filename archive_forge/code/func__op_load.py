from typing import Dict, List
from unittest.mock import patch
import sympy
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, IRNode, Pointwise
from torch._inductor.utils import IndentedBuffer, sympy_str
def _op_load(self, name, index_expr):
    if name == self.accumulator_node_name:
        return '{}'
    elif name in self.aliases:
        return self.aliases[name]
    else:
        raise CUTLASSEVTOpNotImplementedError(f'Operand {name} not found. Auxiliary inputs not supported yet.')