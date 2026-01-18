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
def _repeat_interleave_single_value_repeat_helper(g: jit_utils.GraphContext, self, repeats, dim):
    from torch.onnx.symbolic_opset9 import flatten, unsqueeze
    if not _is_tensor(repeats):
        repeats = g.op('Constant', value_t=torch.LongTensor(repeats))
    const_repeats: bool = _is_constant(repeats)
    reps = _maybe_get_const(repeats, 't')
    if _get_tensor_rank(repeats) == 0:
        repeats = g.op('Reshape', repeats, g.op('Constant', value_t=torch.tensor([1])))
    unsqueezed = unsqueeze(g, self, dim + 1)
    if const_repeats:
        onehot = torch.ones(_get_tensor_rank(unsqueezed), dtype=torch.int64)
        onehot[dim + 1] = reps
        repeats_per_dim = g.op('Constant', value_t=onehot)
    else:
        onehot = g.op('OneHot', unsqueeze(g, dim + 1, 0), g.op('Constant', value_t=torch.tensor(_get_tensor_rank(unsqueezed))), g.op('Concat', g.op('Constant', value_t=torch.tensor([1])), repeats, axis_i=0))
        repeats_per_dim = flatten(g, onehot, 0, 1)
    tiled = g.op('Tile', unsqueezed, repeats_per_dim)
    return flatten(g, tiled, dim, dim + 1)