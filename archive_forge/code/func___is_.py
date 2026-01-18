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
@_onnx_symbolic('aten::__is_')
@_beartype.beartype
def __is_(g: jit_utils.GraphContext, self, other):
    if symbolic_helper._is_none(other):
        if symbolic_helper._is_none(self):
            return g.op('Constant', value_t=torch.BoolTensor([1]))
        return g.op('Constant', value_t=torch.BoolTensor([0]))
    return eq(g, self, other)