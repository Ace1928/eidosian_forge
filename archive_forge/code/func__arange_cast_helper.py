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
def _arange_cast_helper(g: jit_utils.GraphContext, end, start=None, step=None, dtype=None) -> Tuple[_type_utils.JitScalarType, Optional[_C.Value], Optional[_C.Value], Optional[_C.Value]]:

    def _is_all_integral(scalars):
        for scalar in scalars:
            scalar_type = _type_utils.JitScalarType.from_value(scalar, _type_utils.JitScalarType.UNDEFINED)
            if scalar_type != _type_utils.JitScalarType.INT64 and scalar_type != _type_utils.JitScalarType.UNDEFINED:
                return False
        return True
    if dtype is None or (_is_value(dtype) and _is_none(dtype)):
        if _is_all_integral([start, end, step]):
            scalar_type = _type_utils.JitScalarType.INT64
        else:
            scalar_type = _type_utils.JitScalarType.from_dtype(torch.get_default_dtype())
    else:
        assert isinstance(dtype, int)
        scalar_type = _type_utils.JitScalarType(dtype)
    start = g.op('Cast', start, to_i=scalar_type.onnx_type()) if start else None
    end = g.op('Cast', end, to_i=scalar_type.onnx_type()) if end else None
    step = g.op('Cast', step, to_i=scalar_type.onnx_type()) if step else None
    return (scalar_type, end, start, step)