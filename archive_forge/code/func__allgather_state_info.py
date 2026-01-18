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
def _allgather_state_info(fsdp_state: _FSDPState, input_states: Dict[str, Any]) -> List[Dict[str, StateInfo]]:
    """
    Given the ``input_states``, allgather StateInfo for each state. The function
    uses all_gather_object to gather StateInfo so no GPU tensors are sent.
    """
    processed_state_dict: Dict[str, StateInfo] = {}
    gathered_state_info: List[Dict[str, StateInfo]] = [{} for _ in range(fsdp_state.world_size)]
    for fqn, optim_state in input_states.items():
        processed_state = StateInfo({}, {}, {})
        for state_name, value in sorted_items(optim_state):
            if torch.is_tensor(value):
                if value.dim() == 0:
                    processed_state.scalar_tensors[state_name] = value.cpu()
                else:
                    processed_state.tensors[state_name] = _PosDimTensorInfo(value.shape, value.dtype)
            else:
                processed_state.non_tensors[state_name] = value
        processed_state_dict[fqn] = processed_state
    dist.all_gather_object(gathered_state_info, processed_state_dict, group=fsdp_state.process_group)
    return gathered_state_info