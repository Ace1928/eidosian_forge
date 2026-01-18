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
def _init_device_handle(state: _FSDPState, module: nn.Module, ignored_params: Set[nn.Parameter], device_id: Optional[Union[int, torch.device]]) -> _FSDPState:
    """
    Determine device handle used for initializing FSDP.

    If a device is specified by ``device_id``,
    then returns device handle corresponds to that device type. Otherwise, If the
    module is already on a non-CPU device, then the device type is that non-CPU device type.
    If the module is on CPU or meta, then the device type is the current cuda device.

    This method will be called once ignored paramters was determined, as the device handle maybe needed
    for other initialization.
    """
    determined_device = None
    if device_id is not None:
        determined_device = device_id if isinstance(device_id, torch.device) else torch.device(device_id)
    if determined_device is None:
        for param in _get_orig_params(module, ignored_params):
            if param.device.type in {'cpu', 'meta'}:
                continue
            if determined_device is None:
                determined_device = param.device
            elif param.device.type != determined_device.type:
                raise RuntimeError(f'FSDP does not support modules with different device types but got params on {determined_device.type} and {param.device.type}')
        determined_device = determined_device or torch.device('cuda', torch.cuda.current_device())
    state._device_handle = _FSDPDeviceHandle.from_device(determined_device)
    return state