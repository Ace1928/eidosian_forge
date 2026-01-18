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
def _get_compute_device(module: nn.Module, ignored_params: Set[nn.Parameter], device_from_device_id: Optional[torch.device], rank: int) -> torch.device:
    """
    Determine and return this FSDP instance's compute device.

    If a device is
    specified by ``device_id``, then returns that device. Otherwise, If the
    module is already on a non-CPU device, then the compute device is that non-CPU
    device. If the module is on CPU, then the compute device is the current
    device.

    Since this method should be called after materializing the module, any
    non-CPU device should not be meta device. For now, the compute device is
    always a CUDA GPU device with its explicit index.

    Precondition: ``_check_single_device_module()`` and
    ``_move_module_to_device()``.
    """
    param = next(_get_orig_params(module, ignored_params), None)
    if param is not None and param.device.type != 'cpu':
        compute_device = param.device
    elif device_from_device_id is not None and device_from_device_id.type != 'cuda':
        compute_device = device_from_device_id
    else:
        compute_device = torch.device('cuda', torch.cuda.current_device())
    if device_from_device_id is not None and compute_device != device_from_device_id:
        raise ValueError(f'Inconsistent compute device and `device_id` on rank {rank}: {compute_device} vs {device_from_device_id}')
    return compute_device