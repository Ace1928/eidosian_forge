from collections import defaultdict
import logging
import math
from typing import Dict
import torch
import torch.distributed as dist
from . import default_hooks as default
from torch.distributed import distributed_c10d
def _orthogonalize_gram_schmidt(matrices, epsilon=0):
    """
    Applies Gram-Schmidt procedure to orthogonalize a batch of matrices.
    If epsilon is 0, this is equivalent to `torch.qr(matrices, out=(matrices, _))`,
    """
    num_cols = matrices.shape[2]
    for i in range(num_cols):
        col = matrices[:, :, i:i + 1]
        if epsilon == 0:
            try:
                col /= torch.norm(col, dim=1, keepdim=True)
            except ZeroDivisionError:
                logger.error('The matrices to be orthogonalized has at least a column of all 0s. Please set a small value such as 1e-8 as `orthogonalization_epsilon` in PowerSGD state.')
                col.fill_(0.0)
        else:
            col /= torch.norm(col, dim=1, keepdim=True) + epsilon
        if i + 1 < num_cols:
            rest = matrices[:, :, i + 1:]
            rest -= torch.sum(col * rest, dim=1, keepdim=True) * col