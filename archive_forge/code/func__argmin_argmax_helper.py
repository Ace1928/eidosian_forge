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
def _argmin_argmax_helper(g: jit_utils.GraphContext, input: torch._C.Value, dim: torch._C.Value, keepdim: bool, op_name: str):

    def op_wrapper(input, axis_i, keepdims_i):
        if g.opset >= 12:
            return g.op(op_name, input, axis_i=axis_i, keepdims_i=keepdims_i, select_last_index_i=False)
        return g.op(op_name, input, axis_i=axis_i, keepdims_i=keepdims_i)
    if _is_none(dim):
        flattened = _reshape_helper(g, input, g.op('Constant', value_t=torch.tensor([-1])))
        output = op_wrapper(flattened, axis_i=0, keepdims_i=False)
        if keepdim:
            input_shape = g.op('Shape', input)
            input_shape_shape = g.op('Shape', input_shape)
            new_shape = g.op('ConstantOfShape', input_shape_shape, value_t=torch.tensor([1], dtype=torch.int64))
            output = g.op('Reshape', output, new_shape)
        return output
    dim = _parse_arg(dim, 'i')
    return op_wrapper(input, axis_i=dim, keepdims_i=keepdim)