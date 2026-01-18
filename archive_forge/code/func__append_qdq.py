import dataclasses
import itertools
import operator
from typing import Any, Callable, Dict, List, Tuple
import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.subgraph_rewriter import (
import torch.nn.functional as F
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.quantizer import (
from .utils import (
def _append_qdq(x, is_per_channel, kwargs):
    """
    Helper function to append q-dq ops after `x`, using dummy values for the qparams
    and qmin/qmax. We use dummy values here because we match with `ignore_literals=True`
    and will manually replace these values after subgraph rewriting.

    Return the dq node.
    """
    per_channel_axis = 0
    scale = kwargs['scale'] if is_per_channel else 1.0
    zp = kwargs['zero_point'] if is_per_channel else 0
    qmin = -127
    qmax = 127
    dtype = torch.int8
    qd = torch.ops.quantized_decomposed
    if is_per_channel:
        x = qd.quantize_per_channel(x, scale, zp, per_channel_axis, qmin, qmax, dtype)
        x = qd.dequantize_per_channel(x, scale, zp, per_channel_axis, qmin, qmax, dtype)
    else:
        x = qd.quantize_per_tensor(x, scale, zp, qmin, qmax, dtype)
        x = qd.dequantize_per_tensor(x, scale, zp, qmin, qmax, dtype)
    return x