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
def _onnx_opset_unsupported_detailed(op_name: str, current_opset: int, supported_opset: int, reason: str, value: Optional[_C.Value]=None) -> NoReturn:
    message = f'Unsupported: ONNX export of {op_name} in opset {current_opset}. {reason}. Please try opset version {supported_opset}.'
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(message, value)
    raise errors.OnnxExporterError(message)