import torch
from torch.nn import functional as F
from functools import reduce
from typing import Any, List, Optional, Tuple
from .base_data_sparsifier import BaseDataSparsifier
def __get_scatter_folded_mask(self, data, dim, indices, output_size, sparse_block_shape):
    mask = torch.ones_like(data)
    mask.scatter_(dim=dim, index=indices, value=0)
    mask = F.fold(mask, output_size=output_size, kernel_size=sparse_block_shape, stride=sparse_block_shape)
    mask = mask.to(torch.int8)
    return mask