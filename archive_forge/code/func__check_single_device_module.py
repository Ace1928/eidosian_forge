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
def _check_single_device_module(module: nn.Module, ignored_params: Set[nn.Parameter], device_id: Optional[Union[int, torch.device]]) -> None:
    """
    Raise an error if ``module`` has original parameters on multiple devices, ignoring the parameters in ``ignored_params``.

    Thus, after this method, the
    module must be either fully on the CPU or fully on a non-CPU device.
    """
    devices = {param.device for param in _get_orig_params(module, ignored_params)}
    if len(devices) == 2 and torch.device('cpu') in devices:
        if device_id is None:
            raise RuntimeError('To support a module with both CPU and GPU params, please pass in device_id argument.')
    elif len(devices) > 1:
        raise RuntimeError(f'FSDP only supports single device modules but got params on {devices}')