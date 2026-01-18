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
@_onnx_symbolic('aten::_reshape_from_tensor')
@_beartype.beartype
def _reshape_from_tensor(g: jit_utils.GraphContext, input, shape):
    if isinstance(shape, list):
        shape = g.op('Concat', *shape, axis_i=0)
    return reshape(g, input, shape)