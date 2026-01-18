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
def _scatter_helper(g: jit_utils.GraphContext, self, dim, index, src):
    if g.opset <= 10:
        from torch.onnx.symbolic_opset9 import scatter
    else:
        from torch.onnx.symbolic_opset11 import scatter
    return scatter(g, self, dim, index, src)