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
def _init_process_group_state_for_hybrid_shard(state: _FSDPState, process_group: ProcessGroupType, device_mesh: DeviceMesh) -> _FSDPState:
    if device_mesh:
        if _is_valid_hybrid_shard_device_mesh(device_mesh):
            state._device_mesh = device_mesh
            state._inter_node_pg = device_mesh.get_group(mesh_dim=0)
            state.process_group = device_mesh.get_group(mesh_dim=1)
        else:
            raise ValueError(f'Expected device_mesh to have ndim=2 but got {len(device_mesh.get_group())}')
    elif process_group is None:
        default_group = _get_default_group()
        intra_node_group, inter_node_group = _init_intra_and_inter_node_groups(default_group, state._device_handle.device_count())
        state.process_group = intra_node_group
        state._inter_node_pg = inter_node_group
    elif _is_valid_hybrid_shard_pg_type(process_group):
        state.process_group, state._inter_node_pg = process_group
    else:
        raise ValueError(f'Expected process_group to be passed in as either None or Tuple[dist.ProcessGroup, dist.ProcessGroup] but got {type(process_group)}')
    state._inter_node_state = _get_default_comm_hook_state(process_group=state._inter_node_pg)
    return state