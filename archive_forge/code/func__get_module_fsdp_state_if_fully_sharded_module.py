import logging
import traceback
import warnings
import weakref
from enum import auto, Enum
from functools import partial
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._flat_param as flat_param_file
import torch.nn as nn
from torch.distributed._composable_state import _get_module_state, _State
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.utils import _apply_to_tensors
from torch.utils._mode_utils import no_dispatch
from .api import (
def _get_module_fsdp_state_if_fully_sharded_module(module: nn.Module) -> Optional[_FSDPState]:
    state = _get_module_fsdp_state(module)
    if state is None:
        return None
    if state == module:
        return state
    if module in state._fully_sharded_module_to_handle:
        return state
    return None