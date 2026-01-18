from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def _scatter_topk_to_sparse_tensor(top_k_tensor: Tensor, to_be_sparsify_tensor: Tensor, k: int, dim: Optional[int]) -> Tensor:
    """Scatter the topk values of the to_be_sparsify_tensor to a zero tensor of the same shape
    at the top-k indices of the top_k_tensor. This function allows top-k computation with a
    derived tensor from to_be_sparsify_tensor.

    Args:
        top_k_tensor (Tensor):
            The source tensor whose top-k "indices" are taken and used to extract
            the corresponding "values" from the to_be_sparsify_tensor.
        to_be_sparsify_tensor (Tensor):
            The tensor whose values are gathered according to the top-k indices
            of the top_k_tensor, and a zero tensor of same shape is populated with these
            values at those indices and creates the sparse_tensor tensor.
        k (int):
            the value of k for top-k
        dim (Optional[int]):
            dimension for top-k

    Returns:
        (Tensor):
            Returns a sparse_tensor with the same shape as the top_k_tensor and to_be_sparsify_tensor,
            and populated with the values of the to_be_sparsify_tensor at the indices corresponding
            to the top-k indices of the source tensor.
    """
    assert top_k_tensor.shape == to_be_sparsify_tensor.shape, 'top_k_tensor and to_be_sparsify_tensor have different shapes!'
    sparse_tensor = torch.zeros_like(to_be_sparsify_tensor)
    orig_shape = sparse_tensor.shape
    if dim is None and len(orig_shape) > 1:
        sparse_tensor = sparse_tensor.reshape(-1)
        to_be_sparsify_tensor = to_be_sparsify_tensor.reshape(-1)
        top_k_tensor = top_k_tensor.reshape(-1)
        dim = -1
    _, i = top_k_tensor.topk(k, dim=dim)
    return sparse_tensor.scatter(dim, i, to_be_sparsify_tensor.gather(dim, i)).reshape(orig_shape)