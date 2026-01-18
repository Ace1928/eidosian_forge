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
def _reduce_grad(state: _FSDPState, handle: FlatParamHandle) -> None:
    """
    For sharded strategies, this runs gradient reduction, sharded gradient
    accumulation if needed, and the post-reduction callback.
    """
    flat_param = handle.flat_param
    uses_hybrid_sharded_strategy = handle._sharding_strategy in (HandleShardingStrategy.HYBRID_SHARD, HandleShardingStrategy._HYBRID_SHARD_ZERO2)
    unsharded_grad = flat_param.grad.data
    flat_param.grad = None
    padded_unsharded_grad, new_sharded_grad = _get_reduce_scatter_tensors(state, unsharded_grad)
    if state._comm_hook is None:
        _div_if_needed(padded_unsharded_grad, state._gradient_predivide_factor)
        pg = handle._fake_process_group if handle._use_fake_reduce else state.process_group
        dist.reduce_scatter_tensor(new_sharded_grad, padded_unsharded_grad, group=pg)
        if uses_hybrid_sharded_strategy:
            state._all_reduce_stream.wait_stream(state._post_backward_stream)
            with state._device_handle.stream(state._all_reduce_stream):
                _no_dispatch_record_stream(new_sharded_grad, state._all_reduce_stream)
                dist.all_reduce(new_sharded_grad, group=state._inter_node_pg)
                _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
                grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
                _post_reduce_grad_callback(state, handle, grad_to_offload)
                return
        _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
    else:
        state._comm_hook(state._comm_hook_state, padded_unsharded_grad, new_sharded_grad)
    grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
    _post_reduce_grad_callback(state, handle, grad_to_offload)