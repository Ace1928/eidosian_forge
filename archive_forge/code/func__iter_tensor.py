import collections
import functools
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.testing
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrTensors
def _iter_tensor(x_tensor):
    if _is_sparse_any_tensor(x_tensor):

        def get_stride(size):
            dim = len(size)
            tmp = 1
            stride = [0] * dim
            for i in reversed(range(dim)):
                stride[i] = tmp
                tmp *= size[i]
            return stride
        x_nnz = x_tensor._nnz()
        x_size = list(x_tensor.size())
        if x_tensor.layout is torch.sparse_coo:
            x_indices = x_tensor._indices().t()
            x_values = x_tensor._values()
        elif x_tensor.layout is torch.sparse_csr:
            x_indices = torch._convert_indices_from_csr_to_coo(x_tensor.crow_indices(), x_tensor.col_indices()).t()
            x_values = x_tensor.values()
        elif x_tensor.layout is torch.sparse_csc:
            x_indices = torch._convert_indices_from_csr_to_coo(x_tensor.ccol_indices(), x_tensor.row_indices(), transpose=True).t()
            x_values = x_tensor.values()
        elif x_tensor.layout is torch.sparse_bsr:
            x_block_values = x_tensor.values()
            x_blocksize = x_block_values.size()[1:3]
            x_indices = torch._convert_indices_from_csr_to_coo(x_tensor.crow_indices(), x_tensor.col_indices()).repeat_interleave(x_blocksize[0] * x_blocksize[1], 1).mul_(torch.tensor(x_blocksize, device=x_tensor.device).reshape(2, 1)).add_(torch.stack(torch.where(torch.ones(x_blocksize, device=x_tensor.device))).repeat(1, x_nnz)).t()
            x_values = x_block_values.flatten(0, 2)
            x_nnz = x_values.size(0)
        elif x_tensor.layout is torch.sparse_bsc:
            x_block_values = x_tensor.values()
            x_blocksize = x_block_values.size()[1:3]
            x_indices = torch._convert_indices_from_csr_to_coo(x_tensor.ccol_indices(), x_tensor.row_indices(), transpose=True).repeat_interleave(x_blocksize[0] * x_blocksize[1], 1).mul_(torch.tensor(x_blocksize, device=x_tensor.device).reshape(2, 1)).add_(torch.stack(torch.where(torch.ones(x_blocksize, device=x_tensor.device))).repeat(1, x_nnz)).t()
            x_values = x_block_values.flatten(0, 2)
            x_nnz = x_values.size(0)
        else:
            raise NotImplementedError(f'_iter_tensor for {x_tensor.layout} input')
        x_stride = get_stride(x_size)
        x_values = x_values.data
        for i in range(x_nnz):
            x_value = x_values[i]
            for x_idx in product(*[range(m) for m in x_values.size()[1:]]):
                indices = x_indices[i].tolist() + list(x_idx)
                d_idx = sum((indices[k] * x_stride[k] for k in range(len(x_size))))
                yield (x_value, x_idx, d_idx)
    elif x_tensor.layout == torch._mkldnn:
        for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
            x_tensor_dense = x_tensor.to_dense()
            yield (x_tensor_dense, x_idx, d_idx)
    else:
        x_tensor = x_tensor.data
        for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
            yield (x_tensor, x_idx, d_idx)