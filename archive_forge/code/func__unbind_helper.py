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
def _unbind_helper(g: jit_utils.GraphContext, self, dim, _outputs):
    if g.opset < 11:
        from torch.onnx.symbolic_opset9 import unbind
    elif g.opset <= 12:
        from torch.onnx.symbolic_opset11 import unbind
    else:
        from torch.onnx.symbolic_opset13 import unbind
    return unbind(g, self, dim, _outputs)