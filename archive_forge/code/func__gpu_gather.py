import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def _gpu_gather(tensor):
    state = PartialState()
    if is_torch_version('>=', '1.13'):
        gather_op = torch.distributed.all_gather_into_tensor
    else:
        gather_op = torch.distributed._all_gather_base

    def _gpu_gather_one(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if state.backend is not None and state.backend != 'gloo':
            output_tensors = torch.empty(state.num_processes * tensor.numel(), dtype=tensor.dtype, device=state.device)
            gather_op(output_tensors, tensor)
            return output_tensors.view(-1, *tensor.size()[1:])
        else:
            output_tensors = [torch.empty_like(tensor) for _ in range(state.num_processes)]
            torch.distributed.all_gather(output_tensors, tensor)
            return torch.cat(output_tensors, dim=0)
    return recursively_apply(_gpu_gather_one, tensor, error_on_other_type=True)