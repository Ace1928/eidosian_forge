import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
def _get_mask_from_lengths(lengths: Tensor) -> Tensor:
    """Returns a binary mask based on ``lengths``. The ``i``-th row and ``j``-th column of the mask
    is ``1`` if ``j`` is smaller than ``i``-th element of ``lengths.

    Args:
        lengths (Tensor): The length of each element in the batch, with shape (n_batch, ).

    Returns:
        mask (Tensor): The binary mask, with shape (n_batch, max of ``lengths``).
    """
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    return mask