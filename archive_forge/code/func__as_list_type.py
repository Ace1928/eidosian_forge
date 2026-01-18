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
def _as_list_type(jit_type: _C.JitType) -> Optional[_C.ListType]:
    if isinstance(jit_type, _C.ListType):
        return jit_type
    return None