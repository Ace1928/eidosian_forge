from typing import Optional, Tuple
import torch
import torch.distributed
class _ReduceFromModelParallelRegion(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_: torch.Tensor, process_group: torch.distributed.ProcessGroup) -> torch.Tensor:
        all_reduce(input_, process_group=process_group)
        ctx.mark_dirty(input_)
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return (grad_output, None)