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
def _reshape_helper(g: jit_utils.GraphContext, input, shape, allowzero=0):
    shape = _maybe_get_const(shape, 'is')
    if not _is_value(shape):
        shape = g.op('Constant', value_t=torch.LongTensor(shape))
    if g.opset <= 13:
        if allowzero == 1:
            _onnx_opset_unsupported('Reshape with allowzero=1', GLOBALS.export_onnx_opset_version, 14, input)
        return g.op('Reshape', input, shape)
    else:
        return g.op('Reshape', input, shape, allowzero_i=allowzero)