import warnings
import torch
from .core import is_masked_tensor
from .creation import as_masked_tensor, masked_tensor
def _multidim_any(mask, dim, keepdim):
    if isinstance(dim, int):
        return _multidim_any(mask, [dim], keepdim)
    for d in sorted(dim, reverse=True):
        mask = torch.any(mask, dim=d, keepdim=keepdim)
    return mask