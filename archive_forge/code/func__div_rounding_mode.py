from __future__ import annotations
import functools
import sys
import warnings
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.onnx
from torch import _C
from torch.onnx import (
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
@symbolic_helper.parse_args('v', 'v', 's')
@_beartype.beartype
def _div_rounding_mode(g: jit_utils.GraphContext, self, other, rounding_mode):
    if rounding_mode == 'floor':
        return _floor_divide(g, self, other)
    else:
        return opset9._div_rounding_mode(g, self, other, rounding_mode)