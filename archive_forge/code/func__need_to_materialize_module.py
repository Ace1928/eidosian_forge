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
def _need_to_materialize_module(module: nn.Module, ignored_params: Set[nn.Parameter], ignored_modules: Set[nn.Module]) -> Tuple[bool, bool]:
    """
    Return if ``module`` has parameters on meta device and if ``module`` is using torchdistX deferred initialization.

    At most of the returned bools can
    be ``True``. If either is ``True``, then ``module`` needs to be
    materialized.
    """
    managed_params = list(_get_orig_params(module, ignored_params))
    is_meta_module = any((param.is_meta for param in managed_params))
    for submodule in module.modules():
        if submodule in ignored_modules:
            continue
        for buf in submodule.buffers(recurse=False):
            is_meta_module |= buf.is_meta
    is_torchdistX_deferred_init = not is_meta_module and _TORCHDISTX_AVAIL and any((fake.is_fake(param) for param in managed_params))
    return (is_meta_module, is_torchdistX_deferred_init)