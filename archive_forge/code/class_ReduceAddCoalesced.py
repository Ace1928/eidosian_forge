import warnings
import torch
from . import comm
from torch.autograd import Function
from torch._utils import _get_device_index
from typing import List, Optional
class ReduceAddCoalesced(Function):

    @staticmethod
    def forward(ctx, destination, num_inputs, *grads):
        ctx.target_gpus = [grads[i].get_device() for i in range(0, len(grads), num_inputs)]
        grads_ = [grads[i:i + num_inputs] for i in range(0, len(grads), num_inputs)]
        return comm.reduce_add_coalesced(grads_, destination)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None) + Broadcast.apply(ctx.target_gpus, *grad_outputs)