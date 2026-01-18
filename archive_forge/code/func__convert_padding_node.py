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
def _convert_padding_node(input):
    padding = symbolic_helper._maybe_get_const(input, 'is')
    if symbolic_helper._is_value(padding) and symbolic_helper._is_packed_list(padding):
        input_list = symbolic_helper._unpack_list(padding)
        try:
            padding = [symbolic_helper._get_const(v, 'i', 'padding') for v in input_list]
        except Exception:
            return symbolic_helper._onnx_opset_unsupported_detailed('Pad', 9, 11, 'The sizes of the padding must be constant', input)
    return padding