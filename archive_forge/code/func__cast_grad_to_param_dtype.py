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
def _cast_grad_to_param_dtype(state: _FSDPState, sharded_grad: torch.Tensor, param: FlatParameter):
    """
    Casts ``sharded_grad`` back to the full parameter dtype so that the
    optimizer step runs with that dtype. This performs an actual cast if
    1. parameters were in reduced precision during the forward since then
    gradients would be in that reduced precision, or
    2. parameters were not in reduced precision but gradients were in
    reduced precision for communication.
    However, if a low precision communication hook is registered, then this
    dtype cast happens in the hook instead.
    """
    _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
    if not _low_precision_hook_enabled(state) and sharded_grad.dtype != param.dtype:
        low_prec_grad_data = sharded_grad.data
        sharded_grad.data = sharded_grad.data.to(dtype=param.dtype)
        _no_dispatch_record_stream(low_prec_grad_data, state._device_handle.current_stream())