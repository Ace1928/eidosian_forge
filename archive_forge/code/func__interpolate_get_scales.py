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
def _interpolate_get_scales(g: jit_utils.GraphContext, scale_factor, dim):
    offsets = g.op('Constant', value_t=torch.ones(2, dtype=torch.float32))
    scale_factor_rank = _get_tensor_rank(scale_factor)
    if isinstance(scale_factor.type(), _C.ListType) or (scale_factor_rank is not None and scale_factor_rank > 0):
        return g.op('Concat', offsets, scale_factor, axis_i=0)
    else:
        scale_factor = _unsqueeze_helper(g, scale_factor, [0])
        scale_factor = g.op('Cast', scale_factor, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        scales = [scale_factor for i in range(dim - 2)]
    scale_factor = g.op('Concat', offsets, *scales, axis_i=0)
    return scale_factor