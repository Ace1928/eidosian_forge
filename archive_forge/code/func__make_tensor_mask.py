from functools import reduce
from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from .base_sparsifier import BaseSparsifier
def _make_tensor_mask(self, data, input_shape, sparsity_level, sparse_block_shape, mask=None):
    """Creates a tensor-level mask.

        Tensor-level mask is described as a mask, where the granularity of sparsification of the
        smallest patch is the sparse_block_shape. That means, that for a given mask and a
        sparse_block_shape, the smallest "patch" of zeros/ones could be the sparse_block_shape.

        In this context, `sparsity_level` describes the fraction of sparse patches.
        """
    h, w = data.shape[-2:]
    block_h, block_w = sparse_block_shape
    dh = (block_h - h % block_h) % block_h
    dw = (block_w - w % block_w) % block_w
    if mask is None:
        mask = torch.ones(h + dh, w + dw, device=data.device)
    if sparsity_level >= 1.0:
        mask.data = torch.zeros_like(mask)
        return mask
    elif sparsity_level <= 0.0:
        mask.data = torch.ones_like(mask)
        return mask
    values_per_block = reduce(lambda x, y: x * y, sparse_block_shape)
    if values_per_block > 1:
        data = F.avg_pool2d(data[None, None, :], kernel_size=sparse_block_shape, stride=sparse_block_shape, ceil_mode=True)
    data = data.flatten()
    num_blocks = len(data)
    data = data.repeat(1, values_per_block, 1)
    threshold_idx = int(round(sparsity_level * num_blocks))
    threshold_idx = max(0, min(num_blocks - 1, threshold_idx))
    _, sorted_idx = torch.topk(data, k=threshold_idx, dim=2, largest=False)
    mask_reshape = mask.reshape(data.shape)
    self._scatter_fold_block_mask(dim=2, output_shape=(h + dh, w + dw), indices=sorted_idx, block_shape=sparse_block_shape, mask=mask_reshape)
    mask.data = mask_reshape.squeeze().reshape(mask.shape)[:h, :w].contiguous()
    return mask