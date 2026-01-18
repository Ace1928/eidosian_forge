import functools
import torch
import torch.distributed as dist
from typing import Optional
def _low_precision_hook(prec: torch.dtype, state: LowPrecisionState, grad: torch.Tensor, output: torch.Tensor):
    if grad.dtype != prec:
        grad.data = grad.data.to(prec)
    if output is not None:
        if output.dtype != prec:
            output.data = output.data.to(prec)
        reduce_scatter_hook(state, grad, output)
        _decompress(state, output)
    else:
        allreduce_hook(state, grad)
        _decompress(state, grad)