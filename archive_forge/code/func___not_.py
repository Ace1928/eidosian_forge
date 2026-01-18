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
@_onnx_symbolic('aten::__not_')
@_beartype.beartype
def __not_(g: jit_utils.GraphContext, self):
    if not symbolic_helper._is_bool(self):
        raise errors.SymbolicValueError('ONNX export does NOT support exporting bitwise Not for non-boolean input values', self)
    return g.op('Not', self)