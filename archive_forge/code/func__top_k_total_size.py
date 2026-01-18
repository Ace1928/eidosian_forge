from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def _top_k_total_size(tensor: Tensor, topk_dim: Optional[int]) -> int:
    """Get the total size of the input tensor along the topk_dim dimension. When, the
    dimension is None, get the number of elements in the tensor.
    """
    top_k_total_size = tensor.numel() if topk_dim is None else tensor.shape[topk_dim]
    assert top_k_total_size > 0, 'Total size of input tensor along the topk_dim has to be greater than 0.'
    return top_k_total_size