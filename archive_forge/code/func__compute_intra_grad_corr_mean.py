import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
import numpy as np
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.optim import SGD, Optimizer
def _compute_intra_grad_corr_mean(self) -> torch.Tensor:
    """
        Helper function for computing average intra correlation among gradients on different GPUs.
        This should be called under `model.no_sync()` context.
        """
    assert self._world_size > 1, 'Only for distributed training'
    flat_grad = self._gather_flat_grad()
    corr_mean = torch.tensor(0.0).cuda()
    if dist.get_rank() == 0:
        size = flat_grad.numel()
        gathered_tensors = [torch.zeros(size, device=0) for _ in range(self._world_size)]
        dist.gather(flat_grad, gather_list=gathered_tensors, dst=0)
        corr = torch.stack(gathered_tensors).corrcoef()
        corr = corr[torch.triu(torch.ones_like(corr), diagonal=1) == 1]
        corr_mean = corr.mean()
    else:
        dist.gather(flat_grad, gather_list=None, dst=0)
    dist.broadcast(corr_mean, src=0)
    return corr_mean