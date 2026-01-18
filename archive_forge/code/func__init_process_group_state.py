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
def _init_process_group_state(state: _FSDPState, process_group: ProcessGroupType, sharding_strategy: ShardingStrategy, policy: Optional[_Policy], device_mesh: Optional[DeviceMesh]=None) -> _FSDPState:
    if process_group is not None and device_mesh is not None:
        raise ValueError('Cannot pass both process_group and device_mesh at the same time. Please just pass only one of them.')
    is_hybrid_strategy = sharding_strategy in HYBRID_SHARDING_STRATEGIES
    if is_hybrid_strategy:
        if process_group is None and policy is None and (device_mesh is None):
            raise ValueError(f'Manual wrapping with {sharding_strategy}', 'requires explicit specification of process group or device_mesh.')
        else:
            state = _init_process_group_state_for_hybrid_shard(state, process_group, device_mesh)
    elif device_mesh:
        state._device_mesh = device_mesh
        state.process_group = device_mesh.get_group(mesh_dim=0)
    else:
        state.process_group = process_group if process_group is not None else _get_default_group()
    state.rank = state.process_group.rank()
    state.world_size = state.process_group.size()
    data_parallel_world_size = state.world_size
    if is_hybrid_strategy:
        data_parallel_world_size *= state._inter_node_pg.size()
    state._gradient_predivide_factor = default_hooks.DefaultState._get_gradient_predivide_factor(data_parallel_world_size)
    state._gradient_postdivide_factor = data_parallel_world_size / state._gradient_predivide_factor
    return state