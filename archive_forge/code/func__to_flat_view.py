import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
import numpy as np
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.optim import SGD, Optimizer
def _to_flat_view(p: torch.Tensor) -> torch.Tensor:
    """
            Local helper function for _gather_flat_grad.
            Returns a flattened view of the input tensor.
            """
    if p.grad is None:
        return p.new(p.numel()).zero_()
    elif p.grad.is_sparse:
        return p.grad.to_dense().view(-1)
    else:
        return p.grad.view(-1)