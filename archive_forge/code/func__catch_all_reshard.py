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
def _catch_all_reshard(state: _FSDPState) -> None:
    """
    Reshards the parameters that may not have been resharded in the
    post-backward hook. This can happen when a module's output is used in the
    forward pass, meaning that its pre-backward hook runs (unsharding the
    parameter), but the post-backward hook does not run because the output was
    not jused in the loss computation corresponding to this backward pass.
    """
    try:
        if state._handle:
            already_resharded = state._handle.flat_param.data_ptr() == state._handle.flat_param._local_shard.data_ptr() and (not state._handle._skipped_use_sharded_views)
            if already_resharded:
                return
            free_unsharded_flat_param = _should_free_in_backward(state, state._handle)
            _reshard(state, state._handle, free_unsharded_flat_param)
    except Exception as e:
        _p_assert(False, f'Got exception in the catch-all reshard for {state}: {str(e)}', raise_assertion_error=False)
        raise e