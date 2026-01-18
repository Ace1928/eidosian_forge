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
def _init_ignored_module_states(state: _FSDPState, module: nn.Module, ignored_modules: Optional[Iterable[torch.nn.Module]], ignored_states: Union[Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]]=None) -> _FSDPState:
    if ignored_modules is not None and ignored_states is not None:
        raise ValueError('Cannot pass both ignored_modules and ignored_states at the same time. Please just pass ignored_states.')
    ignored_parameters = None
    passed_as_ignored_states = ignored_states is not None
    if passed_as_ignored_states:
        ignored_states_list = list(ignored_states)
        _check_ignored_states(ignored_states_list, True)
    else:
        ignored_states_list = []
        _check_ignored_states(list(ignored_modules) if ignored_modules is not None else [], False)
    if len(ignored_states_list) > 0:
        if isinstance(ignored_states_list[0], nn.Parameter):
            ignored_parameters = ignored_states_list
        else:
            ignored_modules = ignored_states_list
    state._ignored_modules = _get_ignored_modules(module, ignored_modules)
    state._ignored_params = _get_ignored_params(module, state._ignored_modules, ignored_parameters)
    state._ignored_buffer_names = _get_ignored_buffer_names(module, state._ignored_modules)
    return state