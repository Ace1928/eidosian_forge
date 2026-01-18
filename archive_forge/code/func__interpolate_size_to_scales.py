from __future__ import annotations
import functools
import inspect
import sys
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _type_utils, errors
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils
from torch.types import Number
@_beartype.beartype
def _interpolate_size_to_scales(g: jit_utils.GraphContext, input, output_size, dim):
    output_size = _maybe_get_const(output_size, 'is')
    if _is_value(output_size):
        offset = 2
        offsets = g.op('Constant', value_t=torch.ones(offset, dtype=torch.float32))
        dividend = g.op('Cast', output_size, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        divisor = _slice_helper(g, g.op('Shape', input), axes=[0], ends=[sys.maxsize], starts=[offset])
        divisor = g.op('Cast', divisor, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        scale_dims = g.op('Div', dividend, divisor)
        scales = g.op('Concat', offsets, scale_dims, axis_i=0)
    else:
        scales_constant = [1.0 if i < 2 else float(output_size[-(dim - i)]) / float(input.type().sizes()[-(dim - i)]) for i in range(0, dim)]
        scales = g.op('Constant', value_t=torch.tensor(scales_constant, dtype=torch.float32))
    return scales