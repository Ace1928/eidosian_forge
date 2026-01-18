import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed import group, ReduceOp
class _Scatter(Function):

    @staticmethod
    def forward(ctx, src, group, *tensors):
        ctx.src = src
        ctx.group = group
        assert all((t.size() == tensors[0].size() for t in tensors))
        output = torch.zeros_like(tensors[0])
        if dist.get_rank(group=group) == src:
            dist.scatter(output, list(tensors), src, group=group)
        else:
            dist.scatter(output, None, src, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None) + _Gather.apply(ctx.src, ctx.group, grad_output)