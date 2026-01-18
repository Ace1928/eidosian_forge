import torch
from torch.fx import GraphModule
from ..utils import (
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.fx.subgraph_rewriter import replace_pattern
from torch._higher_order_ops.out_dtype import out_dtype
from typing import Optional, Callable, Tuple, Any
from dataclasses import dataclass
from functools import partial
def _reference_quantized_linear(x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, bias_fp32, out_scale, out_zero_point, out_quant_min, out_quant_max):
    x_i8 = torch.ops.aten.clamp(x_i8, x_quant_min, x_quant_max)
    weight_i8 = torch.ops.aten.clamp(weight_i8, weight_quant_min, weight_quant_max)
    x_i16 = x_i8.to(torch.int16)
    weight_i16 = weight_i8.to(torch.int16)
    acc_i32 = out_dtype(torch.ops.aten.linear.default, torch.int32, x_i16 - x_zero_point, weight_i16 - weight_zero_point, None)
    bias_scale = x_scale * weight_scale
    bias_i32 = out_dtype(torch.ops.aten.div.Tensor, torch.int32, bias_fp32, bias_scale)
    acc_i32 = acc_i32 + bias_i32
    acc_i32 = out_dtype(torch.ops.aten.mul.Tensor, torch.int32, acc_i32, x_scale * weight_scale / out_scale) + out_zero_point
    out_i8 = torch.ops.aten.clamp(acc_i32, out_quant_min, out_quant_max).to(torch.int8)
    return out_i8