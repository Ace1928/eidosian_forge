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
def _get_tensor_rank(x: _C.Value) -> Optional[int]:
    if not _is_tensor(x) or x.type() is None:
        return None
    x_type = x.type()
    x_type = typing.cast(_C.TensorType, x_type)
    return x_type.dim()