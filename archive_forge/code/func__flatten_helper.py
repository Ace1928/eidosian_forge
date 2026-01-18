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
def _flatten_helper(g: jit_utils.GraphContext, input, start_dim, end_dim, dim):
    input_size = g.op('Shape', input)
    slice1 = _slice_helper(g, input_size, axes=[0], starts=[0], ends=[start_dim])
    slices = [slice1, g.op('Constant', value_t=torch.tensor([-1], dtype=torch.long))]
    if end_dim < dim - 1:
        slice3 = _slice_helper(g, input_size, axes=[0], starts=[end_dim + 1], ends=[dim])
        slices = [slice1, g.op('Constant', value_t=torch.tensor([-1], dtype=torch.long)), slice3]
    final_shape = g.op('Concat', *slices, axis_i=0)
    from torch.onnx.symbolic_opset9 import _reshape_from_tensor
    return _reshape_from_tensor(g, input, final_shape)