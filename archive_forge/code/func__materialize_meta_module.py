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
def _materialize_meta_module(root_module: nn.Module, device_from_device_id: Optional[torch.device], ignored_modules: Set[nn.Module]):
    materialization_device = device_from_device_id or torch.device(torch.cuda.current_device())
    modules_to_materialize = _get_modules_to_materialize(root_module, ignored_modules)
    try:
        with torch.no_grad():
            for module in modules_to_materialize:
                module_state_iter = itertools.chain(module.parameters(recurse=False), module.buffers(recurse=False))
                has_module_states = len(list(module_state_iter)) > 0
                if has_module_states:
                    module.to_empty(device=materialization_device, recurse=False)
                    module.reset_parameters()
    except BaseException as e:
        warnings.warn(f'Unable to call `reset_parameters()` for module on meta device with error {str(e)}. Please ensure that your module oftype {type(module)} implements a `reset_parameters()` method.')
        raise e