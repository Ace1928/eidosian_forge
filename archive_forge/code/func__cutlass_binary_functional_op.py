from typing import Dict, List
from unittest.mock import patch
import sympy
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, IRNode, Pointwise
from torch._inductor.utils import IndentedBuffer, sympy_str
def _cutlass_binary_functional_op(self, op, a, b):
    return f'{{ /*{op}: */ {a}, {b} }}'