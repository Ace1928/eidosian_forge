import torch
from torch.fx import GraphModule
from ..utils import (
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.fx.subgraph_rewriter import replace_pattern
from torch._higher_order_ops.out_dtype import out_dtype
from typing import Optional, Callable, Tuple, Any
from dataclasses import dataclass
from functools import partial
def _reference_dequantize_per_tensor_int8(x_i8, scale, zero_point, quant_min, quant_max):
    x_i8 = torch.ops.aten.clamp(x_i8, quant_min, quant_max)
    return ((x_i8.to(torch.float32) - zero_point) * scale).to(dtype=torch.float32)