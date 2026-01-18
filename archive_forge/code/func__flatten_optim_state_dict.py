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
def _flatten_optim_state_dict(optim_state_dict: Dict[str, Any], model: nn.Module, use_orig_params: bool=False, optim: Optional[torch.optim.Optimizer]=None, rank0_only: bool=False, group: Optional[dist.ProcessGroup]=None) -> Dict[str, Any]:
    """
    Flattens the full optimizer state dict, still keying by unflattened parameter
    names.

    If ``use_orig_params`` is True, each rank will have all FSDP-managed
    parameters but some of these parameters may be empty due to the sharding.
    For a regular optim.Optimizer, states for those empty parameters will
    not be initialized. So, when aggregating the FQNs across ranks, no assert
    will be raised on a rank even if it does not have all the states -- it is
    valid and FSDP know how to aggregate them. However, FSDP has to ignore
    handling those parameters that are not managed by FSDP and do not exist on
    the local rank -- it is managed by other parallelism and FSDP does not
    know ho to handle/aggregate them.

    Note that ``_flatten_tensor_optim_state`` does not need ``optim`` to
    flatten/shard the state. However, NamedOptimizer and KeyedOptimizer require
    all the states even if the corresponding parameters are empty. To this end,
    ``optim`` will be used to to get the initial state of the empty parameters.
    ``optim`` should only be non-None if the ``optim` is KeyedOptimizer or
    NamedOptimizer.

    Returns:
        Dict[str, Any]: The flattened optimizer state dict.
    """
    SimpleProfiler.reset()
    unflat_osd = optim_state_dict
    if 'state' not in unflat_osd and (not rank0_only):
        raise ValueError('`optim_state_dict` must have the keys "state"to be a valid optimizer state dict')
    param_to_fqns = _get_param_to_fqns(model)
    fqn_to_fsdp_param_info = _get_fqn_to_fsdp_param_info(model)
    fsdp_state = next(iter(fqn_to_fsdp_param_info.values())).state
    if rank0_only:
        unflat_osd = _broadcast_processed_state(fsdp_state, unflat_osd, group=group)
    flat_osd_state: Dict[Union[_OptimStateKey, str], Any] = {}
    unflat_osd_state = unflat_osd['state']
    all_state_keys = set(unflat_osd_state.keys())
    for param, fqns in param_to_fqns.items():
        fqn = fqns[0]
        if fqn not in unflat_osd_state:
            continue
        all_state_keys.difference_update(fqns)
        if rank0_only:
            for fqn in fqns:
                if not unflat_osd_state[fqn]:
                    continue
                for state_name in unflat_osd_state[fqn].keys():
                    unflat_osd_state[fqn][state_name] = _broadcast_state(fsdp_state, unflat_osd_state[fqn][state_name], group=group)
            fqn = fqns[0]
        if fqn in fqn_to_fsdp_param_info:
            fsdp_param_info = fqn_to_fsdp_param_info[fqn]
            if use_orig_params:
                with SimpleProfiler.profile(SimpleProfiler.Type.RESHARDING):
                    flat_state = _shard_orig_param_state(fsdp_param_info, fqn, unflat_osd_state[fqn])
            else:
                flat_state = _flatten_optim_state(fsdp_param_info, unflat_osd_state, fqns)
            key = _OptimStateKey(tuple(fqns), True)
            if flat_state:
                flat_osd_state[key] = flat_state
            elif use_orig_params:
                assert len(fqns) == 1, f'use_orig_params is True but there are multiple FQNs, {fqns}.'
                if optim is not None:
                    state = optim.state.get(param, None)
                    if state is not None:
                        flat_osd_state[key] = copy.deepcopy(state)
                    else:
                        warnings.warn(f'optim_state[{key}] is not on rank{fsdp_state.rank}.')
            else:
                raise RuntimeError(f'The state of {key} is empty. This should happen when use_orig_params=True.')
        else:
            assert len(fqns) == 1
            key = _OptimStateKey(tuple(fqns), False)
            flat_osd_state[key] = copy.copy(unflat_osd_state[fqn])
        if rank0_only:
            for fqn in fqns:
                if not unflat_osd_state[fqn]:
                    continue
                for state_name, param_state in list(unflat_osd_state[fqn].items()):
                    if fsdp_state.rank > 0:
                        del unflat_osd_state[fqn][state_name]
                    else:
                        unflat_osd_state[fqn][state_name] = unflat_osd_state[fqn][state_name].cpu()
    for key in all_state_keys:
        user_state = unflat_osd_state[key]
        if isinstance(user_state, torch.Tensor) and rank0_only and use_orig_params:
            user_state = _broadcast_state(fsdp_state, user_state, group=group)
        flat_osd_state[key] = copy.copy(user_state)
    SimpleProfiler.dump_and_reset('FSDP _flatten_optim_state_dict() profiling: ')
    if 'param_groups' in unflat_osd:
        flat_osd_param_groups = copy.deepcopy(unflat_osd['param_groups'])
        return {'state': flat_osd_state, 'param_groups': flat_osd_param_groups}
    else:
        return {'state': flat_osd_state}