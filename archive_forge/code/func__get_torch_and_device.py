import numpy as np
from ..parser import convert_to_valid_einsum_chars
from ..sharing import to_backend_cache_wrap
def _get_torch_and_device():
    global _TORCH_DEVICE
    global _TORCH_HAS_TENSORDOT
    if _TORCH_DEVICE is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _TORCH_DEVICE = (torch, device)
        _TORCH_HAS_TENSORDOT = hasattr(torch, 'tensordot')
    return _TORCH_DEVICE