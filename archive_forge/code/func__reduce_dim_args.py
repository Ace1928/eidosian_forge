import warnings
import torch
from .core import is_masked_tensor
from .creation import as_masked_tensor, masked_tensor
def _reduce_dim_args(input, dim, keepdim=False, dtype=None):
    return (input, dim, keepdim, dtype)