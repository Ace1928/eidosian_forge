import warnings
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch import Tensor
from torch.masked import as_masked_tensor, is_masked_tensor, MaskedTensor
from . import _docs
from torch._prims_common import corresponding_real_dtype
from torch import sym_float
def _sparse_coo_where(mask: Tensor, input: Tensor, fill_value: Tensor) -> Tensor:
    """Sparse variant of torch.where. Supports sparse COO and hybrid sparse COO tensors.

    _sparse_coo_where implements the following invariant:

      _sparse_coo_where(mask, input, fill_value).to_dense(fill_value) ==
        torch.where(mask.to_dense(), input.to_dense(), torch.full(input.shape, fill_value))

    where `a == b` means `assertEqual(a, b)`, mask is boolean sparse
    tensor, and `to_dense(fill_value)` is like `to_dense()` except
    that the unspecified elements are mapped to `fill_value` rather
    than to `0`.

    Returns a sparse COO tensor with the following features:

    - all specified elements correspond to masked-in elements that
      have the values of the input tensor. If there exists a masked-in
      element (as specified by mask) that is not specified in the
      input, in the result tensor, the corresponding element has value
      0. In the dense part of the sparse tensor, the masked-out
      elements are replaced with fill_value.

    - all unspecified elements correspond to masked-out elements.
    """
    assert input.layout == torch.sparse_coo
    assert mask.layout == input.layout
    assert mask.shape == input.shape
    assert mask.dense_dim() == input.dense_dim()
    input = input.coalesce()
    input_flat_indices = _sparse_coo_flatten_indices(input.indices(), input.shape[:input.sparse_dim()])
    mask_flat_indices = _sparse_coo_flatten_indices(mask.indices(), mask.shape[:mask.sparse_dim()])
    if mask.dense_dim() > 0:
        mask_values = _any(mask.values(), tuple(range(1, input.sparse_dim() + 1)), False)
    else:
        mask_values = mask.values()
    maskin_flat_indices = mask_flat_indices[mask_values.nonzero()[:, 0]]

    def intersection(i1, i2):
        union, counts = torch.cat([i1, i2]).unique(return_counts=True)
        return (union, torch.where(counts.gt(1)))

    def minus(i1, i2):
        union, counts = torch.cat([i1, i2]).unique(return_counts=True)
        return intersection(union[torch.where(counts.eq(1))], i1)

    def _apply(a):
        obj, w = a
        return obj[w]
    maskin_input_flat_indices = _apply(intersection(maskin_flat_indices, input_flat_indices))
    _, w = intersection(input_flat_indices, maskin_input_flat_indices)
    where_input_indices = input.indices()[(slice(None),) + w]
    where_input_values = input.values()[w]
    if mask.dense_dim() > 0:
        _, w1 = intersection(mask_flat_indices, maskin_input_flat_indices)
        where_mask_values = mask.values()[w1]
        where_input_values = torch.where(where_mask_values, where_input_values, fill_value)
    maskin_zero_flat_indices = _apply(minus(maskin_flat_indices, maskin_input_flat_indices))
    _, w = intersection(mask_flat_indices, maskin_zero_flat_indices)
    where_zero_indices = mask.indices()[(slice(None),) + w]
    n = where_zero_indices.size(1)
    if n == 0:
        result = torch.sparse_coo_tensor(where_input_indices, where_input_values, input.shape)
        return result._coalesced_(True)
    where_indices = torch.cat([where_input_indices, where_zero_indices], dim=1)
    where_values = torch.cat([where_input_values, where_input_values.new_zeros((n,) + where_input_values.shape[1:])])
    result = torch.sparse_coo_tensor(where_indices, where_values, input.shape)
    return result.coalesce()