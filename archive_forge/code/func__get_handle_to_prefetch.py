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
def _get_handle_to_prefetch(state: _FSDPState, current_handle: FlatParamHandle) -> FlatParamHandle:
    """
    Returns a :class:`list` of the handles keys to prefetch for the next
    module(s), where ``current_handle`` represents the current module.

    "Prefetching" refers to running the unshard logic early (without
    synchronization), and the "next" modules depend on the recorded execution
    order and the current training state.
    """
    training_state = _get_training_state(current_handle)
    valid_training_states = (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.BACKWARD_POST, HandleTrainingState.FORWARD)
    _p_assert(training_state in valid_training_states, f'Prefetching is only supported in {valid_training_states} but currently in {training_state}')
    eod = state._exec_order_data
    target_handle: Optional[FlatParamHandle] = None
    if training_state == HandleTrainingState.BACKWARD_PRE and state.backward_prefetch == BackwardPrefetch.BACKWARD_PRE or (training_state == HandleTrainingState.BACKWARD_POST and state.backward_prefetch == BackwardPrefetch.BACKWARD_POST):
        target_handle_candidate = eod.get_handle_to_backward_prefetch(current_handle)
        if target_handle_candidate and target_handle_candidate._needs_pre_backward_unshard and (not target_handle_candidate._prefetched):
            target_handle = target_handle_candidate
        else:
            target_handle = None
    elif training_state == HandleTrainingState.FORWARD and state.forward_prefetch:
        target_handle_candidate = eod.get_handle_to_forward_prefetch(current_handle)
        if target_handle_candidate and target_handle_candidate._needs_pre_forward_unshard and (not target_handle_candidate._prefetched):
            target_handle = target_handle_candidate
        else:
            target_handle = None
    return target_handle