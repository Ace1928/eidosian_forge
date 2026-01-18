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
def _common_pre_state_dict_hook(module: nn.Module, fsdp_state: _FSDPState) -> None:
    """Performs the pre-state_dict tasks shared by all state_dict types."""
    if fsdp_state._device_handle.is_available():
        fsdp_state._device_handle.synchronize()
    _lazy_init(fsdp_state, module)
    if fsdp_state._is_root:
        _reset_flat_param_grad_info_if_needed(fsdp_state._all_handles)