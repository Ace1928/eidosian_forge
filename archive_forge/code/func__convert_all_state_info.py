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
def _convert_all_state_info(fsdp_param_info: FSDPParamInfo, gathered_state_info: List[Dict[str, StateInfo]], input_states: Dict[str, Any], output_states: Dict[str, Dict[str, Any]]) -> Tuple[Optional[torch.dtype], Dict[str, List[Optional[torch.Tensor]]]]:
    """
    Given the ``gathered_state_info`` and ``input_states``, the API converted
    the StateInfo into the original state if the state is not a non-scalar
    tensor. For a multi-dimensional tensor, the local state will be stored in
    ``state_buffer`` in a correct order for later allgather purpose.
    """
    state_buffers: Dict[str, List[Optional[torch.Tensor]]] = {}
    for fqn, gathered_state in output_states.items():
        state_info = [s[fqn] for s in gathered_state_info]
        all_tensor_states = sorted({n for state in state_info for n in state.tensors.keys()})
        empty_ranks: Set[int] = set()
        dtype: Optional[torch.dtype] = None
        for state_name in all_tensor_states:
            numels = []
            _empty_ranks: Set[int] = set()
            for rank, object_state in enumerate(state_info):
                numels.append(0)
                info = object_state.tensors.get(state_name, None)
                if info is not None:
                    numels[-1] = info.shape.numel()
                    if not dtype:
                        dtype = info.dtype
                    else:
                        assert dtype == info.dtype
                if numels[-1] == 0:
                    _empty_ranks.add(rank)
            assert not empty_ranks or empty_ranks == _empty_ranks
            empty_ranks = _empty_ranks
            if state_name not in state_buffers:
                state_buffers[state_name] = [None for _ in fsdp_param_info.param_indices]
            local_state = input_states[fqn].get(state_name, None)
            if local_state is not None:
                local_state = local_state.to(fsdp_param_info.state.compute_device)
            state_buffers[state_name][fsdp_param_info.param_indices[fqn]] = local_state
        for rank, object_state in enumerate(state_info):
            if rank in empty_ranks:
                continue
            for name, non_tensor_value in object_state.non_tensors.items():
                curr_non_tensor_value = gathered_state.get(name, None)
                assert curr_non_tensor_value is None or curr_non_tensor_value == non_tensor_value, f'Rank {rank} has different values for {name}: {non_tensor_value}.' + f' Other ranks: {curr_non_tensor_value}'
                gathered_state[name] = non_tensor_value
            for name, scalar_tensor_value in object_state.scalar_tensors.items():
                curr_scalar_tensor_value = gathered_state.get(name, None)
                assert curr_scalar_tensor_value is None or torch.equal(scalar_tensor_value, curr_scalar_tensor_value), f'Rank {rank} has different values for {name}: {scalar_tensor_value}.' + f' Other ranks: {curr_scalar_tensor_value}'
                gathered_state[name] = scalar_tensor_value
    return (dtype, state_buffers)