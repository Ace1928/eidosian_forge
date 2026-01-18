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
def _maybe_get_const(value: Optional[Union[_C.Value, torch.Tensor, Number, Sequence]], descriptor: _ValueDescriptor):
    if isinstance(value, _C.Value) and _is_onnx_constant(value):
        return _parse_arg(value, descriptor)
    return value