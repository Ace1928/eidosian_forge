from __future__ import annotations
import functools
import sys
import warnings
from typing import Optional, Sequence
import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx import (
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::_dim_arange')
@symbolic_helper.parse_args('v', 'i')
@_beartype.beartype
def _dim_arange(g: jit_utils.GraphContext, like, dim):
    like_shape = g.op('Shape', like)
    stop = g.op('Gather', like_shape, g.op('Constant', value_t=torch.tensor(dim)), axis_i=0)
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.op('_caffe2::Range', stop)
    return arange(g, stop, 4, None, None, None)