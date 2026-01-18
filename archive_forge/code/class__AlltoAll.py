import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed import group, ReduceOp
class _AlltoAll(Function):

    @staticmethod
    def forward(ctx, group, out_tensor_list, *tensors):
        ctx.group = group
        ctx.input_tensor_size_list = [tensors[i].size() for i in range(dist.get_world_size(group=group))]
        my_rank = dist.get_rank(group=group)
        tensors = tuple((t.contiguous() for t in tensors))
        if dist.get_backend(group=group) is dist.Backend.GLOO:
            for i in range(dist.get_world_size(group=group)):
                to_send = None
                if i == my_rank:
                    to_send = list(tensors)
                dist.scatter(out_tensor_list[i], to_send, i, group=group)
        else:
            dist.all_to_all(out_tensor_list, list(tensors), group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        tensor_list = [torch.empty(size, device=grad_outputs[0].device, dtype=grad_outputs[0].dtype) for size in ctx.input_tensor_size_list]
        return (None, None) + _AlltoAll.apply(ctx.group, tensor_list, *grad_outputs)