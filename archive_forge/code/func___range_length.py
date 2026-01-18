from __future__ import annotations
import builtins
import functools
import math
import sys
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch import _C
from torch.onnx import _constants, _deprecation, _type_utils, errors, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
from torch.types import Number
@_onnx_symbolic('aten::__range_length')
@_beartype.beartype
def __range_length(g: jit_utils.GraphContext, lo, hi, step):
    sub = g.op('Sub', hi, lo)
    div = g.op('Ceil', true_divide(g, sub, step))
    return g.op('Cast', div, to_i=_C_onnx.TensorProtoDataType.INT64)