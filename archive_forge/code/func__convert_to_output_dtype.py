from typing import Dict, List
from unittest.mock import patch
import sympy
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, IRNode, Pointwise
from torch._inductor.utils import IndentedBuffer, sympy_str
def _convert_to_output_dtype(self, a):
    return f'cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<identity_op, ElementD, ElementAcc, RoundStyle>,{a}>'