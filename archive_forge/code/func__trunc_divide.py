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
@_beartype.beartype
def _trunc_divide(g: jit_utils.GraphContext, self, other):
    out = g.op('Div', self, other)
    out = g.op('Cast', out, to_i=_C_onnx.TensorProtoDataType.INT64)
    scalar_type = _type_utils.JitScalarType.from_value(self, _type_utils.JitScalarType.UNDEFINED)
    if scalar_type != _type_utils.JitScalarType.UNDEFINED:
        if not symbolic_helper._is_fp(self) and symbolic_helper._is_fp(other):
            out = g.op('Cast', out, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        else:
            out = g.op('Cast', out, to_i=scalar_type.onnx_type())
    else:
        out = g.op('Cast', out, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    return out