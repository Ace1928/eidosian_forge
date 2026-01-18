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
def _index_fill_reshape_helper(g: jit_utils.GraphContext, self, dim, index):
    from torch.onnx.symbolic_opset9 import expand
    if g.opset <= 10:
        from torch.onnx.symbolic_opset9 import scatter
    else:
        from torch.onnx.symbolic_opset11 import scatter
    if self.type().dim() is None:
        return _unimplemented('index_fill', 'input rank not accessible')
    self_dim = self.type().dim()
    dim_value = _parse_arg(dim, 'i')
    if dim_value < 0:
        dim_value += self_dim
    unsqueezed_index = _unsqueeze_helper(g, index, [i for i in range(self_dim) if i != dim_value])
    expanded_index_shape = scatter(g, g.op('Shape', self), 0, _unsqueeze_helper(g, dim, [0]), g.op('Shape', index))
    expanded_index = expand(g, unsqueezed_index, expanded_index_shape, None)
    return (expanded_index_shape, expanded_index)