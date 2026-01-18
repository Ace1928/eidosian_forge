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
@_beartype.beartype
def _lt_impl(g: jit_utils.GraphContext, input, other):
    if symbolic_helper._is_bool(input) and symbolic_helper._is_bool(other):
        input = g.op('Cast', input, to_i=_C_onnx.TensorProtoDataType.INT32)
        other = g.op('Cast', other, to_i=_C_onnx.TensorProtoDataType.INT32)
    return g.op('Less', input, other)