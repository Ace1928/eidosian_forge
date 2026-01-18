import warnings
import torch
from .core import is_masked_tensor
from .creation import as_masked_tensor, masked_tensor
def _apply_reduction(fn, *args, **kwargs):
    if fn in NATIVE_REDUCE_MAP:
        return NATIVE_REDUCE_MAP[fn](*args, **kwargs)
    if fn in TORCH_REDUCE_MAP:
        return TORCH_REDUCE_MAP[fn](*args, **kwargs)
    if fn in TENSOR_REDUCE_MAP:
        return TENSOR_REDUCE_MAP[fn](*args, **kwargs)
    return NotImplemented