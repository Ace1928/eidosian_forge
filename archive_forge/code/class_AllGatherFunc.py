from typing import Optional
import torch
from torch import Tensor
from torch.distributed import ProcessGroup
class AllGatherFunc(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = all_gather_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_input, _ = reduce_scatter_raw(grad_output, ctx.process_group)
        return (grad_input, None)