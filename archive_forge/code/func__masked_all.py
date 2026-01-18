import warnings
import torch
from .core import is_masked_tensor
from .creation import as_masked_tensor, masked_tensor
def _masked_all(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 1:
        return _masked_all_all(args[0], mask=kwargs['mask'])
    return _masked_all_dim(*args, **kwargs)