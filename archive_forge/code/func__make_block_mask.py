from functools import reduce
from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from .base_sparsifier import BaseSparsifier
def _make_block_mask(self, data, sparse_block_shape, zeros_per_block, mask=None):
    """Creates a block-level mask.

        Block-level mask is described as a mask, where the granularity of sparsification of the
        largest patch is the sparse_block_shape. That means that for a given mask and a
        sparse_block_shape, the sparsity is computed only within a patch of a size sparse_block_shape.

        In this context the `zeros_per_block` describes the number of zeroed-out elements within a patch.
        """
    h, w = data.shape[-2:]
    block_h, block_w = sparse_block_shape
    dh = (block_h - h % block_h) % block_h
    dw = (block_w - w % block_w) % block_w
    values_per_block = reduce(lambda x, y: x * y, sparse_block_shape)
    if mask is None:
        mask = torch.ones((h + dh, w + dw), device=data.device)
    if values_per_block == zeros_per_block:
        mask.data = torch.zeros_like(mask)
        return mask
    padded_data = torch.ones(h + dh, w + dw, dtype=data.dtype, device=data.device)
    padded_data.fill_(torch.nan)
    padded_data[:h, :w] = data
    unfolded_data = F.unfold(padded_data[None, None, :], kernel_size=sparse_block_shape, stride=sparse_block_shape)
    mask_reshape = mask.reshape(unfolded_data.shape)
    _, sorted_idx = torch.topk(unfolded_data, k=zeros_per_block, dim=1, largest=False)
    self._scatter_fold_block_mask(dim=1, indices=sorted_idx, output_shape=padded_data.shape, block_shape=sparse_block_shape, mask=mask_reshape)
    mask.data = mask_reshape.squeeze().reshape(mask.shape).contiguous()
    return mask