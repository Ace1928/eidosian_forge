from __future__ import annotations
import functools
import sys
import warnings
from typing import Optional, Sequence
import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx import (
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::__lshift_')
@_beartype.beartype
def __lshift_(g: jit_utils.GraphContext, self, other):
    if _type_utils.JitScalarType.from_value(other, _type_utils.JitScalarType.UNDEFINED) != _type_utils.JitScalarType.from_value(self):
        other = g.op('Cast', other, to_i=_type_utils.JitScalarType.from_value(self).onnx_type())
    if _type_utils.JitScalarType.from_value(self, _type_utils.JitScalarType.UNDEFINED) == _type_utils.JitScalarType.UINT8:
        return g.op('BitShift', self, other, direction_s='LEFT')
    two = g.op('Constant', value_t=torch.tensor(2, dtype=torch.float32))
    if not symbolic_helper._is_fp(self):
        other = g.op('Cast', other, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    two_pow = g.op('Pow', two, other)
    two_pow = g.op('Cast', two_pow, to_i=_type_utils.JitScalarType.from_value(self).onnx_type())
    lshift = g.op('Mul', self, two_pow)
    return lshift