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
def _is_split_static(split_size_or_sizes, _outputs):
    if _outputs is None:
        return False
    if _is_value(split_size_or_sizes) and split_size_or_sizes.node().kind() != 'onnx::Constant':
        return False
    return True