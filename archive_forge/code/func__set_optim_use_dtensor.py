import copy
import functools
import logging
import warnings
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed.checkpoint._state_dict_utils import _gather_state_dict
from torch.distributed.distributed_c10d import _get_pg_default_device
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._debug_utils import SimpleProfiler
from torch.distributed.fsdp._flat_param import FlatParameter, FlatParamHandle
from torch.distributed.fsdp._fsdp_extensions import (
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.fsdp.api import (
from torch.utils._pytree import tree_map_only
@no_type_check
def _set_optim_use_dtensor(fsdp_state: _FSDPState, state_dict_settings: StateDictSettings) -> None:
    if getattr(fsdp_state, '_device_mesh', None):
        state_dict_type = state_dict_settings.state_dict_type
        if state_dict_type == StateDictType.LOCAL_STATE_DICT:
            raise RuntimeError('Found state_dict_type LOCAL_STATE_DICT.', 'DeviceMesh is not compatible with LOCAL_STATE_DICT.', 'Please set state_dict_type to SHARDED_STATE_DICT to get DTensor state_dict.')
        elif state_dict_type == StateDictType.FULL_STATE_DICT:
            logger.warning('Found both state_dict_type FULL_STATE_DICT and device_mesh. Please set state_dict_type to SHARDED_STATE_DICT to get DTensor state_dict.')
        else:
            state_dict_settings.optim_state_dict_config._use_dtensor = True