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
def _check_ignored_states(ignored_states: List[Any], passed_as_ignored_states: bool) -> None:
    """
    Check that the ignored states are uniformly parameters or uniformly modules.

    We may remove this check in the future if we permit mixing.
    """
    if len(ignored_states) == 0:
        return
    if passed_as_ignored_states:
        all_params = all((isinstance(state, nn.Parameter) for state in ignored_states))
        all_modules = all((isinstance(state, nn.Module) for state in ignored_states))
        if not all_params and (not all_modules):
            sorted_types = sorted({type(state) for state in ignored_states}, key=repr)
            raise ValueError(f'ignored_states expects all nn.Parameter or all nn.Module list elements but got types {sorted_types}')
    elif not all((isinstance(state, nn.Module) for state in ignored_states)):
        sorted_types = sorted({type(state) for state in ignored_states}, key=repr)
        raise ValueError(f'ignored_modules expects nn.Module list elements but got types {sorted_types}')