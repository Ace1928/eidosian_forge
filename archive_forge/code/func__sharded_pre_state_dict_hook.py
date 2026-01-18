import contextlib
import logging
import math
import warnings
from typing import Any, Callable, cast, Dict, Generator, Iterator, no_type_check, Tuple
import torch
import torch.distributed as dist
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as checkpoint_wrapper
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import _mesh_resources
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._debug_utils import SimpleProfiler
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.fsdp.api import (
from torch.distributed.utils import _replace_by_prefix
from ._fsdp_extensions import (
from ._unshard_param_utils import _unshard_fsdp_state_params, FLAT_PARAM
def _sharded_pre_state_dict_hook(fsdp_state: _FSDPState, module: nn.Module, *args, **kwargs) -> None:
    """
    Hook that runs before model.state_dict() is called. Check
    ``_full_pre_load_state_dict_hook`` for the detail.
    """
    if _has_fsdp_params(fsdp_state, module) and (not _module_handle(fsdp_state, module).uses_sharded_strategy):
        raise RuntimeError('``sharded_state_dict`` can only be used when parameters are flatten and sharded.')
    _common_pre_state_dict_hook(module, fsdp_state)
    _common_unshard_pre_state_dict_hook(module, fsdp_state, offload_to_cpu=False, rank0_only=False)