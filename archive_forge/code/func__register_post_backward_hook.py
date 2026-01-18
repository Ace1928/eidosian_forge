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
def _register_post_backward_hook(state: _FSDPState, handle: Optional[FlatParamHandle]) -> None:
    """
    Registers post-backward hooks on the ``FlatParameter`` s'
    ``AccumulateGrad`` objects to reshard and to reduce-scatter gradients.

    The ``AccumulateGrad`` object represents the last function that finalizes
    the ``FlatParameter`` 's gradient, so it only runs after its entire
    gradient computation has finished.

    We register the post-backward hook only once in the *first* forward that a
    ``FlatParameter`` participates in. This relies on the ``AccumulateGrad``
    object being preserved through multiple forwards.

    NOTE: We follow this heuristic to prefer the *first* forward to target the
    parameter mixed precision case, where there are *separate*
    ``AccumulateGrad`` objects across the different forwards. (Without
    parameter mixed precision, the ``AccumulateGrad`` objects are the same.) If
    we instead prefer the *last* forward, then the hook runs early.
    """
    if not torch.is_grad_enabled():
        return
    if not handle:
        return
    flat_param = handle.flat_param
    already_registered = hasattr(flat_param, '_post_backward_hook_state')
    if already_registered or not flat_param.requires_grad:
        return
    temp_flat_param = flat_param.expand_as(flat_param)
    _p_assert(temp_flat_param.grad_fn is not None, 'The `grad_fn` is needed to access the `AccumulateGrad` and register the post-backward hook')
    acc_grad = temp_flat_param.grad_fn.next_functions[0][0]
    assert acc_grad is not None
    hook_handle = acc_grad.register_hook(functools.partial(_post_backward_hook, state, handle))
    flat_param._post_backward_hook_state = (acc_grad, hook_handle)