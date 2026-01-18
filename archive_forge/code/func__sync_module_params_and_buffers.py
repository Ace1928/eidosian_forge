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
def _sync_module_params_and_buffers(module: nn.Module, params: List[nn.Parameter], process_group: dist.ProcessGroup) -> None:
    """
    Synchronize module states (i.e. parameters ``params`` and all not-yet-synced buffers) by broadcasting from rank 0 to all ranks.

    Precondition: ``sync_module_states == True`` and ``self.process_group`` has
    been set.
    """
    module_states: List[torch.Tensor] = []
    for buffer in module.buffers():
        if not getattr(buffer, FSDP_SYNCED, False):
            setattr(buffer, FSDP_SYNCED, True)
            detached_buffer = buffer.detach()
            if is_traceable_wrapper_subclass(detached_buffer):
                attrs, _ = detached_buffer.__tensor_flatten__()
                inner_buffers = [getattr(detached_buffer, attr) for attr in attrs]
                module_states.extend(inner_buffers)
            else:
                module_states.append(detached_buffer)
    for param in params:
        detached_param = param.detach()
        if is_traceable_wrapper_subclass(detached_param):
            attrs, _ = detached_param.__tensor_flatten__()
            inner_params = [getattr(detached_param, attr) for attr in attrs]
            module_states.extend(inner_params)
        else:
            module_states.append(detached_param)
    _check_module_states_for_sync_module_states(module_states)
    _sync_params_and_buffers(process_group, module_states, PARAM_BROADCAST_BUCKET_SIZE, src=0)