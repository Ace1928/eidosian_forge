import torch
from torch.fx import GraphModule
from ..utils import (
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.fx.subgraph_rewriter import replace_pattern
from torch._higher_order_ops.out_dtype import out_dtype
from typing import Optional, Callable, Tuple, Any
from dataclasses import dataclass
from functools import partial
def _qdq_quantized_max_pool2d(x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, out_scale, out_zero_point, out_quant_min, out_quant_max):
    kernel_size = 1
    stride = 1
    padding = 0
    dilation = 1
    ceil_mode = False
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8)
    out_fp32, _ = torch.ops.aten.max_pool2d_with_indices.default(x_fp32, kernel_size, stride, padding, dilation, ceil_mode)
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(out_fp32, out_scale, out_zero_point, out_quant_min, out_quant_max, torch.int8)
    return out_i8