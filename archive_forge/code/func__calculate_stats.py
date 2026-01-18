from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup
from fairscale.internal import torch_version
from fairscale.nn.checkpoint import is_checkpointing, is_recomputing
def _calculate_stats(input: Tensor, eps: float, process_group: ProcessGroup) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    dim = [d for d in range(input.ndim) if d != 1]
    count = torch.full((1,), input.numel() // input.size(1), device=input.device, dtype=input.dtype)
    total_count = count.clone()
    all_reduce_handle = dist.all_reduce(total_count, group=process_group, async_op=True)
    mean = torch.mean(input, dim=dim, keepdim=True)
    meansqr = torch.mean(input * input, dim=dim, keepdim=True)
    vec = torch.cat([mean, meansqr])
    all_reduce_handle.wait()
    vec = vec * (count / total_count)
    dist.all_reduce(vec, group=process_group)
    mean, meansqr = vec.chunk(2)
    var = meansqr - mean * mean
    invstd = torch.rsqrt(var + eps)
    return (mean, var, invstd, total_count)