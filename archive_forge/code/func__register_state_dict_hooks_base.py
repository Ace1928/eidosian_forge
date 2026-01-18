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
@no_type_check
def _register_state_dict_hooks_base(state: _FSDPState, hook_registration_fn_name: str, hook: Callable, hook_registration_fn_kwargs: Dict[str, Any]) -> None:
    """Registers ``hook`` using ``hook_registration_fn``."""
    if not _is_composable(state):
        getattr(state, hook_registration_fn_name)(hook, **hook_registration_fn_kwargs)
    else:
        handle = state._handle
        if handle:
            getattr(handle._fully_sharded_module, hook_registration_fn_name)(hook, **hook_registration_fn_kwargs)