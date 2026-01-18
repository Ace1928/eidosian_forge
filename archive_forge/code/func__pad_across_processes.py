import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def _pad_across_processes(tensor, dim=0, pad_index=0, pad_first=False):
    if getattr(tensor, 'is_nested', False):
        warnings.warn('Cannot pad nested tensors without more information. Leaving unprocessed.', CannotPadNestedTensorWarning)
        return tensor
    if dim >= len(tensor.shape):
        return tensor
    size = torch.tensor(tensor.shape, device=tensor.device)[None]
    sizes = gather(size).cpu()
    max_size = max((s[dim] for s in sizes))
    if max_size == tensor.shape[dim]:
        return tensor
    old_size = tensor.shape
    new_size = list(old_size)
    new_size[dim] = max_size
    new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
    if pad_first:
        indices = tuple((slice(max_size - old_size[dim], max_size) if i == dim else slice(None) for i in range(len(new_size))))
    else:
        indices = tuple((slice(0, old_size[dim]) if i == dim else slice(None) for i in range(len(new_size))))
    new_tensor[indices] = tensor
    return new_tensor