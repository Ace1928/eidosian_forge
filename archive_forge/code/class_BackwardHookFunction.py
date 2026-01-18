import torch
import torch.distributed as dist
from torch.autograd.function import Function
class BackwardHookFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        ctx.mark_non_differentiable(*[arg for arg in args if not arg.requires_grad])
        return args

    @staticmethod
    def backward(ctx, *args):
        return args