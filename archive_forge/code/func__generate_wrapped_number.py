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
def _generate_wrapped_number(g: jit_utils.GraphContext, scalar):
    """Creates a wrapped number based on https://github.com/pytorch/pytorch/issues/9515.

    A Tensor is a considered a "wrapped number" if it is
    auto-wrapped from a C++ or Python number type. Integer types are
    wrapped as 0-dim int64 tensors and floating-point types are
    wrapped as 0-dim double tensors.

    The input to this function is constant value. If the data type
    is a floating point type, it is converted to a 0-dim double
    tensor, else it is converted to a 0-dim tensor of its original type
    """
    assert not isinstance(scalar, torch.Tensor)
    if isinstance(scalar, float):
        return g.op('Constant', value_t=torch.tensor(scalar, dtype=torch.double))
    return g.op('Constant', value_t=torch.tensor(scalar))