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
def _register_pre_backward_hooks(state: _FSDPState, module: nn.Module, outputs: Any, handle: FlatParamHandle) -> None:
    """
    Registers pre-backward hooks on the tensors that require gradients in the
    forward pass outputs ``outputs``, which were computed using the
    ``FlatParameter`` s of ``handles``.

    Args:
        module (nn.Module): Fully sharded module (see [Note: Fully Sharded
            Module]).

    Returns:
        Forward pass outputs with pre-backward hooks registered to tensors that
        require gradients.
    """
    if not torch.is_grad_enabled():
        return outputs
    if state._is_root:
        state._post_backward_callback_queued = False
    if handle:
        handle._needs_pre_backward_unshard = False
        handle._ran_pre_backward_hook = False

    def _register_hook(t: torch.Tensor) -> torch.Tensor:
        if t.requires_grad:
            t.register_hook(functools.partial(_pre_backward_hook, state, module, handle))
            if handle:
                handle._needs_pre_backward_unshard = True
        return t
    return _apply_to_tensors(_register_hook, outputs)