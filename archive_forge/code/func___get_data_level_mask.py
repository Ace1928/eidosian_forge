import torch
from torch.nn import functional as F
from functools import reduce
from typing import Any, List, Optional, Tuple
from .base_data_sparsifier import BaseDataSparsifier
def __get_data_level_mask(self, data, sparsity_level, sparse_block_shape):
    height, width = (data.shape[-2], data.shape[-1])
    block_height, block_width = sparse_block_shape
    dh = (block_height - height % block_height) % block_height
    dw = (block_width - width % block_width) % block_width
    data_norm = F.avg_pool2d(data[None, None, :], kernel_size=sparse_block_shape, stride=sparse_block_shape, ceil_mode=True)
    values_per_block = reduce(lambda x, y: x * y, sparse_block_shape)
    data_norm = data_norm.flatten()
    num_blocks = len(data_norm)
    data_norm = data_norm.repeat(1, values_per_block, 1)
    _, sorted_idx = torch.sort(data_norm, dim=2)
    threshold_idx = round(sparsity_level * num_blocks)
    sorted_idx = sorted_idx[:, :, :threshold_idx]
    mask = self.__get_scatter_folded_mask(data=data_norm, dim=2, indices=sorted_idx, output_size=(height + dh, width + dw), sparse_block_shape=sparse_block_shape)
    mask = mask.squeeze(0).squeeze(0)[:height, :width]
    return mask