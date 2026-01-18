from functools import reduce
from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from .base_sparsifier import BaseSparsifier
def _scatter_fold_block_mask(self, output_shape, dim, indices, block_shape, mask=None, input_shape=None, device=None):
    """Creates patches of size `block_shape` after scattering the indices."""
    if mask is None:
        assert input_shape is not None
        mask = torch.ones(input_shape, device=device)
    mask.scatter_(dim=dim, index=indices, value=0)
    mask.data = F.fold(mask, output_size=output_shape, kernel_size=block_shape, stride=block_shape)
    return mask