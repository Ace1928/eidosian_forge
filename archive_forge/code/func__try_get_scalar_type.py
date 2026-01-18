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
def _try_get_scalar_type(*args) -> Optional[_type_utils.JitScalarType]:
    for arg in args:
        scalar_type = _type_utils.JitScalarType.from_value(arg, _type_utils.JitScalarType.UNDEFINED)
        if scalar_type != _type_utils.JitScalarType.UNDEFINED:
            return scalar_type
    return None