import torch
from torch.fx import GraphModule
from ..utils import (
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.fx.subgraph_rewriter import replace_pattern
from torch._higher_order_ops.out_dtype import out_dtype
from typing import Optional, Callable, Tuple, Any
from dataclasses import dataclass
from functools import partial
def _reference_quantized_add(x_i8, x_scale, x_zero_point, y_i8, y_scale, y_zero_point, out_scale, out_zero_point, quant_min, quant_max):
    """
    # How to Derive the formula for out_i8 based on x_i8 and y_i8
    # (since quantized add takes x_i8, y_i8 and their quantization parameters, and produce an out_i8)

    # out_i8 is quantized output, we can write down the formula for it first:
out_i8 = out_f32 / out_scale + out_zero_point           (1)

    # then out_fp32 is computed from x_f32 + y_f32, and the x_fp32 and y_fp32 are the dequantized x_i8 and y_i8
    out_f32 = x_f32 + y_f32           (2)
    x_fp32 = (x_i8 - x_zero_point) * x_scale         (3)
    y_fp32 = (y_i8 - y_zero_point) * y_scale         (4)

    # applying the above fomula to the out_i8 equation we can get the following:
    out_i8 = out_fp32 / out_scale + out_zero_point             # (1)
       = (x_f32 + y_f32) / out_scale + out_zero_point      # applying (2) to substitute out_fp32 with x_fp32 + y_fp32
       = ((x_i8 - x_zero_point) * x_scale + (y_i8 - y_zero_point) * y_scale) / out_scale + out_zero_point  # apply (3) and (4)
    """
    x_i32 = x_i8.to(torch.int32)
    y_i32 = y_i8.to(torch.int32)
    x_i32 = torch.round(x_scale / out_scale * (x_i32 - x_zero_point)).to(torch.int32)
    y_i32 = torch.round(y_scale / out_scale * (y_i32 - y_zero_point)).to(torch.int32)
    out_i32 = x_i32 + y_i32 + out_zero_point
    quant_min = -128
    quant_max = 127
    out_i8 = torch.ops.aten.clamp(out_i32, quant_min, quant_max).to(torch.int8)
    return out_i8