import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.nn.functional as Fn
from xformers.components.attention import (
from xformers.components.attention.core import (
def _kmeans_spherical(self, x: torch.Tensor, K: int, num_iters=10):
    """
        Arguments:
            x: (B, N, D)
        """
    B, N, D = x.size()
    assert K <= N, f'{K} > {N}'
    c = x[:, torch.randperm(N, device=x.device)[:K], :].clone()
    with profiler.record_function('kmeans_spherical'):
        counts = c.new_zeros(B, K)
        ones = x.new_ones((B, N))
        for _ in range(num_iters):
            D_ij = torch.matmul(x, c.transpose(-2, -1))
            cl = D_ij.argmax(dim=-1, keepdim=True).long()
            c.zero_()
            c.scatter_add_(-2, cl.repeat(1, 1, D), x)
            counts.fill_(1e-06)
            counts.scatter_add_(-1, cl.squeeze(-1), ones)
            c.divide_(counts.unsqueeze(-1))
            c = Fn.normalize(c, p=2, dim=-1)
    return c