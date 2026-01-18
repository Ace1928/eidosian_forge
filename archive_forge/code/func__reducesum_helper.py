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
def _reducesum_helper(g: jit_utils.GraphContext, input, axes_i=None, keepdims_i=1, noop_with_empty_axes_i=0):
    keepdims_i = _maybe_get_const(keepdims_i, 'i')
    if g.opset >= 13:
        if axes_i:
            if not _is_value(axes_i):
                axes_i = g.op('Constant', value_t=torch.tensor(axes_i, dtype=torch.long))
            return g.op('ReduceSum', input, axes_i, keepdims_i=keepdims_i, noop_with_empty_axes_i=noop_with_empty_axes_i)
        return g.op('ReduceSum', input, keepdims_i=keepdims_i, noop_with_empty_axes_i=noop_with_empty_axes_i)
    else:
        return g.op('ReduceSum', input, axes_i=axes_i, keepdims_i=keepdims_i)