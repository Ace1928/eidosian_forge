import contextlib
import functools
import logging
import os
import warnings
from enum import auto, Enum
from itertools import accumulate, chain
from typing import (
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.fsdp._common_utils import (
from torch.distributed.utils import _alloc_storage, _free_storage, _p_assert
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]
from ._fsdp_extensions import _ext_post_unflatten_transform, _ext_pre_flatten_transform
def _use_sharded_flat_param(self) -> None:
    """Switches to using the sharded flat parameter."""
    flat_param = self.flat_param
    if self._use_orig_params:
        in_forward = self._training_state == HandleTrainingState.FORWARD
        skip_use_sharded_views = torch.is_grad_enabled() and in_forward and (self._sharding_strategy in NO_RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES)
        if skip_use_sharded_views:
            unsharded_flat_param = flat_param.data
    if self._offload_params:
        device = flat_param._local_shard.device
        _p_assert(device == torch.device('cpu'), f'Expects the local shard to be on CPU but got {device}')
    flat_param.data = flat_param._local_shard
    if self._use_orig_params:
        if skip_use_sharded_views:
            self._unsharded_flat_param_for_skipped_views = unsharded_flat_param
        else:
            self._use_sharded_views()
        if in_forward and (not self._skipped_use_sharded_views):
            accumulated_grad_in_no_sync = flat_param.grad is not None and self.uses_sharded_strategy and (flat_param.grad.shape == flat_param._unpadded_unsharded_size)
            if accumulated_grad_in_no_sync:
                self._use_unsharded_grad_views()
            else:
                self._use_sharded_grad_views()