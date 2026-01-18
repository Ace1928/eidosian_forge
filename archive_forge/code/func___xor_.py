from __future__ import annotations
import builtins
import functools
import math
import sys
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch import _C
from torch.onnx import _constants, _deprecation, _type_utils, errors, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
from torch.types import Number
@_onnx_symbolic('aten::__xor_')
@_beartype.beartype
def __xor_(g: jit_utils.GraphContext, input, other):
    if not symbolic_helper._is_bool(input):
        raise errors.SymbolicValueError('ONNX export does NOT support exporting bitwise XOR for non-boolean input values', input)
    if not symbolic_helper._is_bool(other):
        raise errors.SymbolicValueError('ONNX export does NOT support exporting bitwise XOR for non-boolean input values', other)
    return g.op('Xor', input, other)