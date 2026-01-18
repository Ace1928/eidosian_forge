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
def _interpolate_get_scales_and_mode(g: jit_utils.GraphContext, input, size, scale_factor, mode, align_corners):
    mode = _maybe_get_const(mode, 's')
    if 'linear' in mode:
        mode = 'linear'
    if 'cubic' in mode:
        mode = 'cubic'
    _interpolate_warning(mode)
    align_corners = _maybe_get_const(align_corners, 'b')
    if isinstance(align_corners, bool) and align_corners:
        return _unimplemented('interpolate', 'align_corners == True')
    if not input.type().dim():
        return _unimplemented('interpolate', 'missing input shape')
    dim = input.type().dim()
    if not _is_none(scale_factor):
        scale_factor = _interpolate_get_scales(g, scale_factor, dim)
    elif not _is_none(size):
        if not _is_packed_list(size):
            is_scalar = _maybe_get_const(size, 't').dim() == 0
            if is_scalar:
                size = _unsqueeze_helper(g, size, [0])
                size = [size for i in range(dim - 2)]
                size = g.op('Concat', *size, axis_i=0)
        scale_factor = _interpolate_size_to_scales(g, input, size, dim)
    else:
        return _unimplemented('interpolate', 'Both size and scales are None in __interpolate')
    return (scale_factor, mode)