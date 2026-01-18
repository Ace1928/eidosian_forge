from __future__ import annotations
import functools
from typing import Optional
import torch
from torch.onnx import _constants, _type_utils, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
@_beartype.beartype
def _attention_scale(g: jit_utils.GraphContext, query: torch._C.Value) -> torch._C.Value:
    """Calculate the scale factor for the attention result.

    Args:
        query: Tensor of shape [..., L, E]

    Returns:
        Scalar scale factor := 1 / math.sqrt(query.size(-1))
    """
    query_shape = g.op('Shape', query)
    query_shape_last = g.op('Slice', query_shape, g.op('Constant', value_t=torch.tensor([-1], dtype=torch.int64)), g.op('Constant', value_t=torch.tensor([_constants.INT64_MAX], dtype=torch.int64)))
    embedding_size = g.op('Cast', query_shape_last, to_i=_type_utils.JitScalarType.from_value(query).onnx_type())
    const_one = g.op('Constant', value_t=torch.tensor([1.0], dtype=torch.float))
    scale = g.op('Div', const_one, g.op('Sqrt', embedding_size))
    scale = g.op('Cast', scale, to_i=_type_utils.JitScalarType.from_value(query).onnx_type())
    return scale