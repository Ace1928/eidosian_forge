import functools
import logging
from enum import auto, Enum
from typing import Any, Callable, Dict, List, no_type_check, Optional, Set, Tuple
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.graph import register_multi_grad_hook
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._flat_param import (
from torch.distributed.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES
from torch.distributed.fsdp.api import BackwardPrefetch
from torch.distributed.utils import (
from torch.utils import _pytree as pytree
@no_type_check
@torch.no_grad()
def _post_backward_hook(state: _FSDPState, handle: FlatParamHandle, *unused: Any):
    """
    Reduce-scatters the gradient of ``handle`` 's ``FlatParameter``.

    Precondition: The ``FlatParameter`` 's ``.grad`` attribute contains the
    unsharded gradient for the local batch.

    Postcondition:
    - If using ``NO_SHARD``, then the ``.grad`` attribute is the reduced
    unsharded gradient.
    - Otherwise, the ``_saved_grad_shard`` attribute is the reduced sharded
    gradient (accumulating with any existing gradient).
    """
    _log_post_backward_hook(state, handle, log)
    flat_param = handle.flat_param
    flat_param._post_backward_called = True
    with torch.autograd.profiler.record_function('FullyShardedDataParallel._post_backward_hook'):
        _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
        _p_assert(handle._training_state in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.BACKWARD_POST), f'Expects `BACKWARD_PRE` or `BACKWARD_POST` state but got {handle._training_state}')
        handle._training_state = HandleTrainingState.BACKWARD_POST
        if flat_param.grad is None:
            return
        if flat_param.grad.requires_grad:
            raise RuntimeError('FSDP does not support gradients of gradients')
        _post_backward_reshard(state, handle)
        if not state._sync_gradients:
            if handle._use_orig_params:
                handle._use_unsharded_grad_views()
            return
        state._post_backward_stream.wait_stream(state._device_handle.current_stream())
        with state._device_handle.stream(state._post_backward_stream):
            autograd_computed_grad = flat_param.grad.data
            if not _low_precision_hook_enabled(state) and flat_param.grad.dtype != handle._reduce_dtype and (not handle._force_full_precision):
                flat_param.grad.data = flat_param.grad.to(handle._reduce_dtype)
            if handle.uses_sharded_strategy:
                _reduce_grad(state, handle)
            else:
                _reduce_grad_no_shard(state, handle)
            _no_dispatch_record_stream(autograd_computed_grad, state._post_backward_stream)