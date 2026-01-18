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
def _should_free_in_backward(state: _FSDPState, handle: FlatParamHandle) -> bool:
    """
    Returns whether FSDP should free the unsharded flat parameter in the
    post-backward or not.
    """
    if not handle.uses_sharded_strategy:
        return False
    return state._sync_gradients or handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES