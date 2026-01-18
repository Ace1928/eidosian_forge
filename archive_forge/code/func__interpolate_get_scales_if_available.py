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
def _interpolate_get_scales_if_available(g: jit_utils.GraphContext, scales):
    available_scales = _maybe_get_const(scales[0], 'fs') != -1 and (not _is_none(scales[0]))
    if not available_scales:
        return None
    offsets = g.op('Constant', value_t=torch.ones(2, dtype=torch.float32))
    scales_list = g.op('Constant', value_t=torch.tensor(_maybe_get_const(scales[0], 'fs')))
    scales = g.op('Concat', offsets, scales_list, axis_i=0)
    return scales