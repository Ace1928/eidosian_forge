from __future__ import annotations
import functools
import sys
from typing import Optional, Tuple
import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import (
from torch.onnx._internal import _beartype, jit_utils, registration
@_beartype.beartype
def _einsum_helper(g: jit_utils.GraphContext, equation, tensors):
    if not tensors:
        raise RuntimeError('Einsum inputs are empty.')
    if symbolic_helper._is_bool(tensors[0]):
        tensors = [g.op('Cast', tensor, to_i=_C_onnx.TensorProtoDataType.INT64) for tensor in tensors]
        return g.op('Cast', g.op('Einsum', *tensors, equation_s=equation), to_i=_C_onnx.TensorProtoDataType.BOOL)
    else:
        return g.op('Einsum', *tensors, equation_s=equation)