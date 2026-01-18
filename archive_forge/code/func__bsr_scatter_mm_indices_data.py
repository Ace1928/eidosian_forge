import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
@lru_cache(maxsize=TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE)
def _bsr_scatter_mm_indices_data(indices_format, M, K, N, Ms, Ks, nbatches, SPLIT_N, compressed_sparse_tensor_as_key):
    bsr = compressed_sparse_tensor_as_key.obj
    assert bsr is not None
    crow_indices, col_indices = (bsr.crow_indices(), bsr.col_indices())
    device = crow_indices.device
    indices_dtype = torch.int32
    if indices_format == 'bsr_strided_mm_compressed':
        Ns = N // SPLIT_N
        q_offsets_lst = []
        b = torch.arange(SPLIT_N, dtype=indices_dtype, device=device) * Ns
        for m in range(M // Ms):
            r0 = crow_indices[m].item()
            r1 = crow_indices[m + 1].item()
            if r1 == r0:
                continue
            q_offsets_lst.append((col_indices[r0:r1] * (Ks * N)).repeat(SPLIT_N) + b.repeat_interleave(r1 - r0))
        q_offsets = torch.cat(q_offsets_lst)
        crow_indices_diff = crow_indices.diff()
        non_zero_row_indices = crow_indices_diff.nonzero()
        a = non_zero_row_indices * (Ms * N)
        r_offsets = (a + b).view(-1)
        c_indices = crow_indices
        nnz_per_row = crow_indices_diff[non_zero_row_indices].repeat_interleave(SPLIT_N)
        nnz_per_row, indices = nnz_per_row.sort(descending=True, stable=True)
        r_offsets = r_offsets[indices]
        return (indices_format, c_indices, r_offsets, q_offsets)
    elif indices_format == 'bsr_strided_mm':
        Ns = N // SPLIT_N
        p_offsets_lst = []
        q_offsets_lst = []
        b = torch.arange(SPLIT_N, dtype=indices_dtype, device=device) * Ns
        for m in range(M // Ms):
            r0 = crow_indices[m].item()
            r1 = crow_indices[m + 1].item()
            if r1 == r0:
                continue
            p_offsets_lst.append(torch.arange(r0, r1, dtype=indices_dtype, device=device).repeat(SPLIT_N))
            q_offsets_lst.append((col_indices[r0:r1] * (Ks * N)).repeat(SPLIT_N) + b.repeat_interleave(r1 - r0))
        q_offsets = torch.cat(q_offsets_lst)
        crow_indices_diff = crow_indices.diff()
        non_zero_row_indices = crow_indices_diff.nonzero()
        a = non_zero_row_indices * (Ms * N)
        r_offsets = (a + b).view(-1)
        c_indices = torch.cat((crow_indices[:1], torch.cumsum(crow_indices_diff[non_zero_row_indices].repeat_interleave(SPLIT_N), 0)))
        p_offsets = torch.cat(p_offsets_lst)
        return (indices_format, c_indices, r_offsets, p_offsets, q_offsets)
    elif indices_format == 'scatter_mm':
        Ns = Ms
        c_indices = [0]
        pq_offsets = []
        for b in range(nbatches):
            for m in range(M // Ms):
                r0 = crow_indices[m].item()
                r1 = crow_indices[m + 1].item()
                for n in range(N // Ns):
                    c_indices.append(c_indices[-1] + r1 - r0)
                    for t in range(r1 - r0):
                        p = r0 + t
                        q = (col_indices[p].item() + b * (K // Ks)) * (N // Ns) + n
                        pq_offsets.append([p, q])
        return (indices_format, torch.tensor(c_indices, dtype=indices_dtype, device=device), torch.tensor(pq_offsets, dtype=indices_dtype, device=device))
    else:
        raise ValueError(f'Invalid indices_format={indices_format!r}. Expected bsr_strided_mm_compressed|bsr_strided_mm|scatter_mm')