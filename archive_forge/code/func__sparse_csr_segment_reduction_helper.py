import warnings
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch import Tensor
from torch.masked import as_masked_tensor, is_masked_tensor, MaskedTensor
from . import _docs
from torch._prims_common import corresponding_real_dtype
from torch import sym_float
def _sparse_csr_segment_reduction_helper(op, mask_input: Tensor, dims: Tuple[int, ...], keepdim: bool, dtype: Optional[DType]=None) -> Tensor:
    assert keepdim, 'reduction operations on CSR tensors with keepdim=False is unsupported'
    reduce = op.__name__
    valid_reductions = ['sum', 'prod', 'mean', 'amax', 'amin']
    if reduce not in valid_reductions:
        raise ValueError(f'op must be one of {' '.join(valid_reductions)}, but got {reduce} instead')
    device = mask_input.device
    output_dtype = dtype
    values, crow_indices, col_indices = (mask_input.values(), mask_input.crow_indices(), mask_input.col_indices())
    if values.dtype != output_dtype:
        values = values.to(output_dtype)
    if len(dims) == 0:
        return mask_input
    if len(dims) == 1:
        if dims[0] == 0:
            new_col_indices, scatter_indices = torch.unique(col_indices, return_inverse=True)
            new_nnz = new_col_indices.shape[0]
            new_crow_indices = torch.tensor([0, new_nnz])
            new_values = values.new_empty(new_col_indices.shape)
            new_values.scatter_reduce_(0, scatter_indices, values, reduce, include_self=False)
            new_shape = [1, mask_input.size(1)]
        else:
            assert dims[0] == 1, 'Sparse CSR tensors are 2D and only support reduction along dim 0 or 1.'
            new_crow_indices = torch.cat((crow_indices.new_zeros(1), torch.cumsum(torch.diff(crow_indices) != 0, 0)), 0)
            new_nnz = new_crow_indices[-1]
            new_col_indices = col_indices.new_zeros(new_nnz)
            new_values = torch._segment_reduce(values, reduce, offsets=crow_indices)
            new_shape = [mask_input.size(0), 1]
    else:
        assert len(dims) == 2
        nnz = min(1, values.numel())
        if nnz == 1:
            op_kwargs = {'keepdim': True, 'dtype': output_dtype}
            if reduce in ['amax', 'amin']:
                del op_kwargs['dtype']
            new_values = op(values, 0, **op_kwargs)
        else:
            new_values = torch.empty(0, dtype=output_dtype)
        new_col_indices = col_indices.new_zeros(nnz)
        new_crow_indices = torch.tensor([0, nnz])
        new_shape = [1, nnz]
    return torch.sparse_csr_tensor(new_crow_indices, new_col_indices, new_values, new_shape, dtype=output_dtype, device=device)