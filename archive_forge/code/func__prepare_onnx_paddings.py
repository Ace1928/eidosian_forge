from __future__ import annotations
import functools
import sys
import warnings
from typing import Optional, Sequence
import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx import (
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
@_beartype.beartype
def _prepare_onnx_paddings(g: jit_utils.GraphContext, input, pad):
    """Generate paddings in ONNX order based on pad in pytorch.

    Args:
        input: the input tensor.
        pad: the paddings in pytorch.
            The order is dim_n_begin, dim_n_end, dim_n-1_begin, dim_n-1_end, ..., dim_m_begin, dim_m_end,
            where m is in range [0, n].
    """
    if not symbolic_helper._is_packed_list(pad) and symbolic_helper._is_list(pad) and symbolic_helper._is_scalar_list(pad):
        pad = g.op('ConcatFromSequence', pad, axis_i=0, new_axis_i=1)
    pad_len = opset9.size(g, pad, g.op('Constant', value_t=torch.tensor([0])))
    rank = symbolic_helper._get_tensor_rank(input)
    if rank is None:
        rank = g.op('Size', g.op('Shape', input))
    else:
        rank = g.op('Constant', value_t=torch.tensor(rank, dtype=torch.int64))
    extension = g.op('Sub', g.op('Mul', rank, g.op('Constant', value_t=torch.tensor(2, dtype=torch.int64))), pad_len)
    pad = g.op('Cast', pad, to_i=_C_onnx.TensorProtoDataType.INT64)
    paddings = g.op('Concat', pad, g.op('ConstantOfShape', extension, value_t=torch.tensor([0], dtype=torch.int64)), axis_i=0)
    paddings = symbolic_helper._reshape_helper(g, paddings, g.op('Constant', value_t=torch.tensor([-1, 2])))
    paddings = g.op('Transpose', opset10.flip(g, paddings, [0]), perm_i=[1, 0])
    paddings = symbolic_helper._reshape_helper(g, paddings, g.op('Constant', value_t=torch.tensor([-1])))
    padding_c = g.op('Cast', paddings, to_i=_C_onnx.TensorProtoDataType.INT64)
    return padding_c