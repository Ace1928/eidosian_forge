from typing import Optional, Tuple
import torch
import torch.distributed
class _CopyToModelParallelRegion(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_: torch.Tensor, process_group: torch.distributed.ProcessGroup) -> torch.Tensor:
        ctx.process_group = process_group
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        all_reduce(grad_output, process_group=ctx.process_group)
        return (grad_output, None)