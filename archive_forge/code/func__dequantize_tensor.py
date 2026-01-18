import functools
import torch
import torch.distributed as dist
from enum import Enum
def _dequantize_tensor(tensor, qtype, quant_loss=None):
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f'_dequantize_tensor expecting torch.Tensor as input but found {type(tensor)}')
    if qtype == DQuantType.FP16:
        if tensor.dtype != torch.float16:
            raise RuntimeError(f'tensor dtype is {tensor.dtype} while expected to be FP16.')
        elif tensor.dtype == torch.float16 and quant_loss is None:
            return tensor.float()
        else:
            return tensor.float() / quant_loss
    elif qtype == DQuantType.BFP16:
        if tensor.dtype != torch.float16:
            raise RuntimeError(f'tensor dtype is {tensor.dtype} while expected to be FP16.')
        else:
            return torch.ops.quantization._Bfloat16QuantizedToFloat(tensor)
    else:
        raise RuntimeError(f'Quantization type {qtype} is not supported')