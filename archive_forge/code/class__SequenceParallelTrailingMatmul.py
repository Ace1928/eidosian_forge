from typing import Callable, List, Optional, Tuple
import torch
from .common import make_pytorch_cuda_operator
from .differentiable_collectives import (
from .sequence_parallel_fused_ops import (
from .tiled_matmul import tiled_matmul_fwd
class _SequenceParallelTrailingMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fuse: bool, process_group: torch.distributed.ProcessGroup, gathered_input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gathered_input, weight)
        ctx.fuse = fuse
        ctx.process_group = process_group
        scattered_output = sequence_parallel_trailing_matmul_fwd(gathered_input, weight, fuse, process_group)
        return scattered_output

    @staticmethod
    def backward(ctx, grad_scattered_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        gathered_input, weight = ctx.saved_tensors
        grad_gathered_input, grad_weight = sequence_parallel_trailing_matmul_bwd(gathered_input, weight, grad_scattered_output, ctx.fuse, ctx.process_group)
        return (None, None, grad_gathered_input, grad_weight)