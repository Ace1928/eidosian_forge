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
def _repeat_interleave_split_helper(g: jit_utils.GraphContext, self, reps, dim):
    if g.opset <= 12:
        split_out = g.op('Split', self, split_i=[1] * reps, axis_i=dim, outputs=reps)
    else:
        from torch.onnx.symbolic_opset13 import split
        repeats = g.op('Constant', value_t=torch.tensor([1] * reps))
        split_out = split(g, self, repeats, dim, _outputs=reps)
    return split_out if reps > 1 else [split_out]