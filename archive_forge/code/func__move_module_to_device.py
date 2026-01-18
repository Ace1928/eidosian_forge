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
def _move_module_to_device(module: nn.Module, ignored_params: Set[nn.Parameter], ignored_buffers: Set[torch.Tensor], device_from_device_id: Optional[torch.device]) -> None:
    """
    Move ``module`` depending on ``device_from_device_id`` and its current device.

    This includes moving ignored modules' parameters.

    - If ``device_from_device_id`` is not ``None``, then this moves
    ``module`` to the device.
    - If ``device_from_device_id`` is ``None``, then this does not move
    ``module`` but warns the user if it is on CPU.

    Precondition: ``_check_single_device_module()``.
    """
    cpu_device = torch.device('cpu')
    if device_from_device_id is not None:
        queue: Deque[nn.Module] = collections.deque()
        queue.append(module)
        params: List[nn.Parameter] = []
        buffers: List[torch.Tensor] = []
        while queue:
            curr_module = queue.popleft()
            params.extend((param for param in curr_module.parameters(recurse=False) if param.device == cpu_device))
            buffers.extend((buffer for buffer in curr_module.buffers(recurse=False) if buffer.device == cpu_device))
            for submodule in curr_module.children():
                if not isinstance(submodule, fsdp_file.FullyShardedDataParallel):
                    queue.append(submodule)
        params_to_move = [p for p in params if p not in ignored_params]
        bufs_to_move = [p for p in buffers if p not in ignored_buffers]
        _move_states_to_device(params_to_move, bufs_to_move, device_from_device_id)
        return
    param = next(_get_orig_params(module, ignored_params), None)
    if param is not None and param.device == cpu_device:
        _warn_cpu_init()