import torch
from torch.fx import GraphModule
from ..utils import (
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.fx.subgraph_rewriter import replace_pattern
from torch._higher_order_ops.out_dtype import out_dtype
from typing import Optional, Callable, Tuple, Any
from dataclasses import dataclass
from functools import partial
def _reference_quantize_per_tensor_int8(x_fp32, scale, zero_point, quant_min, quant_max):
    x = x_fp32 / scale
    x = torch.round(x)
    x = x.to(dtype=torch.int32)
    x = x + zero_point
    x = torch.clamp(x, quant_min, quant_max)
    x = x.to(dtype=torch.int8)
    return x