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
def _is_all_integral(scalars):
    for scalar in scalars:
        scalar_type = _type_utils.JitScalarType.from_value(scalar, _type_utils.JitScalarType.UNDEFINED)
        if scalar_type != _type_utils.JitScalarType.INT64 and scalar_type != _type_utils.JitScalarType.UNDEFINED:
            return False
    return True