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
@_onnx_symbolic('aten::_convolution_mode')
@symbolic_helper.parse_args('v', 'v', 'v', 'is', 's', 'is', 'i')
@_beartype.beartype
def _convolution_mode(g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups):
    weight_size = symbolic_helper._get_tensor_sizes(weight)
    try:
        kernel_shape = weight_size[2:]
    except Exception:
        kernel_shape = None
    if kernel_shape is None or any((i is None for i in kernel_shape)):
        raise errors.SymbolicValueError('Unsupported: ONNX export of convolution for kernel of unknown shape.', input)
    args = [input, weight]
    if not symbolic_helper._is_none(bias) and symbolic_helper._get_tensor_rank(bias) == 1:
        args.append(bias)
    if padding == 'valid':
        padding = 'VALID'
    elif padding == 'same':
        padding = 'SAME_UPPER'
    kwargs = {'kernel_shape_i': weight_size[2:], 'strides_i': stride, 'auto_pad_s': padding, 'dilations_i': dilation, 'group_i': groups}
    n = g.op('Conv', *args, **kwargs)
    if not symbolic_helper._is_none(bias) and symbolic_helper._get_tensor_rank(bias) != 1:
        return g.op('Add', n, bias)
    else:
        return n