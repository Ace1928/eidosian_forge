import torch
from torch.nn import functional as F
from functools import reduce
from typing import Any, List, Optional, Tuple
from .base_data_sparsifier import BaseDataSparsifier
def __get_block_level_mask(self, data, sparse_block_shape, zeros_per_block):
    height, width = (data.shape[-2], data.shape[-1])
    block_height, block_width = sparse_block_shape
    values_per_block = block_height * block_width
    if values_per_block == zeros_per_block:
        return torch.zeros_like(data, dtype=torch.int8)
    dh = (block_height - height % block_height) % block_height
    dw = (block_width - width % block_width) % block_width
    padded_data = torch.ones(height + dh, width + dw, dtype=data.dtype, device=data.device)
    padded_data = padded_data * torch.nan
    padded_data[0:height, 0:width] = data
    unfolded_data = F.unfold(padded_data[None, None, :], kernel_size=sparse_block_shape, stride=sparse_block_shape)
    _, sorted_idx = torch.sort(unfolded_data, dim=1)
    sorted_idx = sorted_idx[:, :zeros_per_block, :]
    mask = self.__get_scatter_folded_mask(data=unfolded_data, dim=1, indices=sorted_idx, output_size=padded_data.shape, sparse_block_shape=sparse_block_shape)
    mask = mask.squeeze(0).squeeze(0)[:height, :width].contiguous()
    return mask