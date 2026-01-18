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
@_onnx_symbolic('aten::_cast_Half')
@_deprecation.deprecated('2.0', 'the future', 'Avoid using this function and create a Cast node instead')
@_beartype.beartype
def _cast_Half(g: jit_utils.GraphContext, input, non_blocking):
    return g.op('Cast', input, to_i=_C_onnx.TensorProtoDataType.FLOAT16)