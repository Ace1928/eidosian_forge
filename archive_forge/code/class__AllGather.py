import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed import group, ReduceOp
class _AllGather(Function):

    @staticmethod
    def forward(ctx, group, tensor):
        tensor = tensor.contiguous()
        ctx.group = group
        out_tensor_list = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather(out_tensor_list, tensor, group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if dist.get_backend(group=ctx.group) is dist.Backend.NCCL:
            rank = dist.get_rank()
            gx = torch.empty_like(grad_outputs[rank])
            _Reduce_Scatter.apply(ReduceOp.SUM, ctx.group, gx, *grad_outputs)
        else:
            tensor_list = [torch.empty_like(tensor) for tensor in grad_outputs]
            gxs = _AlltoAll.apply(ctx.group, tensor_list, *grad_outputs)
            gx = torch.sum(torch.stack(gxs), dim=0)
        return (None, gx)