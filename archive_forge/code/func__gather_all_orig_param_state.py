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
def _gather_all_orig_param_state(fsdp_param_info: FSDPParamInfo, input_states: Dict[str, Any], shard_state: bool, to_save: bool, cpu_offload: bool) -> Dict[str, Any]:
    """
    Given a optimizer state dict, ``input_states``, which the keys are FQNs to the
    original parameters (not FlatParameters nor parmeter ID), gather all the
    states and unflatten them to the original dimensions. Note that all the
    params referred by the ``input_states`` must be managed by FSDP.
    """
    fsdp_state = fsdp_param_info.state
    if fsdp_state.world_size == 1 or fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        return input_states if to_save else {}
    with SimpleProfiler.profile(SimpleProfiler.Type.RESHARDING):
        with SimpleProfiler.profile(SimpleProfiler.Type.ALLGATHER_OBJ):
            gathered_state_info = _allgather_state_info(fsdp_state, input_states)
        output_states = _allgather_orig_param_states(fsdp_param_info, gathered_state_info, input_states, shard_state, to_save, cpu_offload)
    if to_save:
        for key, idx in fsdp_param_info.param_indices.items():
            if key in output_states:
                continue
            if not fsdp_param_info.param_requires_grad[idx]:
                continue
            raise RuntimeError(f'{key} is not in the output state. The FSDPParamInfo has the param keys {sorted(fsdp_param_info.param_indices.keys())} while the output_states has the param keys {sorted(output_states.keys())}.')
        return output_states
    else:
        return {}