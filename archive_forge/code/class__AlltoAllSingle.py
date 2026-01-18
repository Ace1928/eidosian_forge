import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed import group, ReduceOp
class _AlltoAllSingle(Function):

    @staticmethod
    def forward(ctx, group, output, output_split_sizes, input_split_sizes, input):
        ctx.group = group
        ctx.input_size = input.size()
        ctx.output_split_sizes = input_split_sizes
        ctx.input_split_sizes = output_split_sizes
        dist.all_to_all_single(output, input, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tensor = torch.empty(ctx.input_size, device=grad_output.device, dtype=grad_output.dtype)
        return (None, None, None, None) + (_AlltoAllSingle.apply(ctx.group, tensor, ctx.output_split_sizes, ctx.input_split_sizes, grad_output.contiguous()),)