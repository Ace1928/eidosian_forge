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
def _handle_reduce_dim_none(g: jit_utils.GraphContext, self, op_name):
    rank = _get_tensor_rank(self)
    if rank is not None and any((_get_tensor_dim_size(self, i) == 0 for i in range(rank))):
        return g.op(op_name, self, keepdims_i=1)
    return g.op(op_name, self, keepdims_i=0)