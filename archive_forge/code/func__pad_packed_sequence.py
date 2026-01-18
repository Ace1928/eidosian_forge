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
@_onnx_symbolic('aten::_pad_packed_sequence')
@symbolic_helper.parse_args('v', 'v', 'i', 't', 'v')
@_beartype.beartype
def _pad_packed_sequence(g: jit_utils.GraphContext, data, batch_sizes, batch_first, padding_value, total_length):
    data, lengths = g.op('prim::PadPacked', data, batch_sizes, outputs=2)
    if batch_first:
        data = g.op('Transpose', data, perm_i=[1, 0, 2])
    return (data, lengths)