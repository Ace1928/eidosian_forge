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
def _get_dim_for_cross(x: _C.Value, dim: Optional[int]):
    if dim == -1:
        tensor_rank = _get_tensor_rank(x)
        assert tensor_rank is not None
        return dim + tensor_rank
    if dim is None:
        sizes = _get_tensor_sizes(x)
        assert sizes is not None
        for index, size in enumerate(sizes):
            if size is not None and size == 3:
                return index
    return dim