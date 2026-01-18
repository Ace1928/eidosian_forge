import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed import group, ReduceOp
class _Reduce_Scatter(Function):

    @staticmethod
    def forward(ctx, op, group, tensor, *input_tensor_list):
        ctx.group = group
        tensor = tensor.contiguous()
        input_tensor_list = tuple((t.contiguous() for t in input_tensor_list))
        dist.reduce_scatter(tensor, list(input_tensor_list), op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + _AllGather.apply(ctx.group, grad_output)