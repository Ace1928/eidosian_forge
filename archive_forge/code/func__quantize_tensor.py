import functools
import torch
import torch.distributed as dist
from enum import Enum
def _quantize_tensor(tensor, qtype):
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f'_quantize_tensor expecting torch.Tensor as input but found {type(tensor)}')
    if qtype == DQuantType.FP16:
        return _fp32_to_fp16_with_clamp(tensor)
    elif qtype == DQuantType.BFP16:
        return torch.ops.quantization._FloatToBfloat16Quantized(tensor)
    else:
        raise RuntimeError(f'Quantization type {qtype} is not supported')