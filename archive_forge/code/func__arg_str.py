from typing import Dict, List
from unittest.mock import patch
import sympy
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, IRNode, Pointwise
from torch._inductor.utils import IndentedBuffer, sympy_str
def _arg_str(a):
    if isinstance(a, sympy.Expr):
        return f"{_MAGIC_SYMPY_ERROR_STRING}('{sympy_str(a)}')"
    return str(a)