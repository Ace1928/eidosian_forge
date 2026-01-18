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
def _init_streams(state: _FSDPState) -> None:
    """
    Initializes CUDA streams for overlapping communication, computation, and
    data transfers. The streams should be shared across FSDP instances.
    """
    assert state._is_root
    assert state._device_handle.is_available()
    uses_hybrid_sharding = any((fsdp_state.sharding_strategy in HYBRID_SHARDING_STRATEGIES for fsdp_state in state._all_fsdp_states))
    high_priority = -1 if state.limit_all_gathers and uses_hybrid_sharding else 0
    state._default_stream = state._device_handle.current_stream()
    if state._fsdp_extension is not None:
        state._fsdp_extension.compute_stream = state._default_stream
    state._unshard_stream = state._device_handle.Stream(priority=high_priority)
    state._post_backward_stream = state._device_handle.Stream(priority=high_priority)
    state._pre_unshard_stream = state._device_handle.Stream(priority=high_priority)
    state._all_reduce_stream = state._device_handle.Stream() if uses_hybrid_sharding else state._default_stream