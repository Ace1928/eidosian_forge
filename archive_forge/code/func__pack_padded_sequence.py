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
@_onnx_symbolic('aten::_pack_padded_sequence')
@symbolic_helper.parse_args('v', 'v', 'i')
@_beartype.beartype
def _pack_padded_sequence(g: jit_utils.GraphContext, input, lengths, batch_first):
    if batch_first:
        input = g.op('Transpose', input, perm_i=[1, 0, 2])
    if not lengths.type().isSubtypeOf(torch._C.TensorType.get()):
        raise errors.SymbolicValueError("'lengths' must be a Tensor for ONNX export", input)
    if _type_utils.JitScalarType.from_value(lengths, _type_utils.JitScalarType.UNDEFINED) != _type_utils.JitScalarType.INT:
        lengths = g.op('Cast', lengths, to_i=_C_onnx.TensorProtoDataType.INT32)
    return g.op('prim::PackPadded', input, lengths, outputs=2)