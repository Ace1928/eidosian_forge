import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple
import torch
def _validate_pruning_dim(t, dim):
    """Validate that the pruning dimension is within the bounds of the tensor dimension.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        dim (int): index of the dim along which we define channels to prune
    """
    if dim >= t.dim():
        raise IndexError(f'Invalid index {dim} for tensor of size {t.shape}')