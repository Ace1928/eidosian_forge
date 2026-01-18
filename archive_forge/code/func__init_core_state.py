import collections
import itertools
import os
import warnings
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._exec_order_utils as exec_order_utils
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.distributed.fsdp.fully_sharded_data_parallel as fsdp_file
import torch.nn as nn
from torch.distributed.algorithms._comm_hooks import default_hooks
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._flat_param import (
from torch.distributed.fsdp._limiter_utils import _FreeEventQueue
from torch.distributed.fsdp.api import (
from torch.distributed.fsdp.wrap import _Policy
from torch.distributed.tensor.parallel.fsdp import DTensorExtensions
from torch.distributed.utils import _sync_params_and_buffers
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.hooks import RemovableHandle
@no_type_check
def _init_core_state(state: _FSDPState, sharding_strategy: Optional[ShardingStrategy], mixed_precision: Optional[MixedPrecision], cpu_offload: Optional[CPUOffload], limit_all_gathers: bool, use_orig_params: bool, backward_prefetch_limit: int, forward_prefetch_limit: int) -> _FSDPState:
    if state.world_size == 1:
        if sharding_strategy != ShardingStrategy.NO_SHARD:
            warnings.warn(f'FSDP is switching to use `NO_SHARD` instead of {sharding_strategy or ShardingStrategy.FULL_SHARD} since the world size is 1.')
        sharding_strategy = ShardingStrategy.NO_SHARD
    state.sharding_strategy = sharding_strategy or ShardingStrategy.FULL_SHARD
    state.mixed_precision = mixed_precision or MixedPrecision()
    if mixed_precision is not None:
        torch._C._log_api_usage_once(f'torch.distributed.fsdp.mixed_precision.{str(state.mixed_precision)}')
    state._use_full_prec_in_eval = os.environ.get(_FSDP_USE_FULL_PREC_IN_EVAL, '') == '1'
    state.cpu_offload = cpu_offload or CPUOffload()
    state.limit_all_gathers = limit_all_gathers
    state._use_orig_params = use_orig_params
    state.training_state = TrainingState.IDLE
    state._is_root = None
    state._free_event_queue = _FreeEventQueue()
    state._debug_level = dist.get_debug_level()
    state._exec_order_data = exec_order_utils._ExecOrderData(state._debug_level, backward_prefetch_limit, forward_prefetch_limit)
    _fully_sharded_module_to_handle: Dict[nn.Module, FlatParamHandle] = dict()
    state._fully_sharded_module_to_handle = _fully_sharded_module_to_handle
    _handle: FlatParamHandle = None
    state._handle = _handle
    params: List[FlatParameter] = []
    state.params = params
    return state