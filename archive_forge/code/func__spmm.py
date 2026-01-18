import logging
import torch
from xformers import _is_triton_available
from xformers.ops import masked_matmul
def _spmm(b, layout, values):
    N, nnz, _, block_size = values.shape
    br = b.reshape(b.shape[0], b.shape[1], b.shape[2] // block_size, block_size, b.shape[3])
    h, r, c = layout.nonzero(as_tuple=True)
    temp = values @ br[:, h, c, :]
    linear_idx = h * (b.shape[2] // block_size) + r
    out = torch.zeros(N, b.shape[1] * layout.shape[-2], block_size, b.shape[3], dtype=b.dtype, device=b.device)
    out.index_add_(1, linear_idx.to(b.device), temp)
    out = out.reshape(N, b.shape[1], -1, b.shape[3])
    return out