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
def _unshard(state: _FSDPState, handle: FlatParamHandle, unshard_stream: torch.Stream, pre_unshard_stream: torch.Stream) -> None:
    """
    Unshards the handles in ``handles``. If the handles are in
    :meth:`summon_full_params` and are using mixed precision, then they are
    forced to full precision.

    Postcondition: handle's ``FlatParameter`` 's data is the padded
    unsharded flat parameter on the compute device.
    """
    if not handle:
        return
    with state._device_handle.stream(pre_unshard_stream):
        ran_pre_unshard = handle.pre_unshard()
    if ran_pre_unshard:
        unshard_stream.wait_stream(pre_unshard_stream)
    if state.limit_all_gathers:
        event = state._free_event_queue.dequeue_if_needed()
        if event:
            with torch.profiler.record_function('FullyShardedDataParallel.rate_limiter'):
                event.synchronize()
    with state._device_handle.stream(unshard_stream):
        handle.unshard()
        handle.post_unshard()