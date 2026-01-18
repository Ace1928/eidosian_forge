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
def _convert_state_with_orig_params(all_optim_state_keys: List[_OptimStateKey], optim_state_key_to_param_key: Dict[_OptimStateKey, Union[int, str]], fqn_to_fsdp_param_info: Dict[str, FSDPParamInfo], optim_state_dict: Dict[Union[str, int], Any], to_save: bool, shard_state: bool, cpu_offload: bool=True) -> Dict[str, Any]:
    fsdp_osd_state: Dict[str, Any] = {}
    all_states: Dict[int, Dict[str, Any]] = {}
    for optim_state_key in all_optim_state_keys:
        param_key: Union[str, int, None] = optim_state_key_to_param_key.get(optim_state_key, None)
        if param_key is None and (not optim_state_key.is_fsdp_managed):
            continue
        if optim_state_key.is_fsdp_managed:
            fqn = optim_state_key.unflat_param_names[0]
            fsdp_param_info = fqn_to_fsdp_param_info.get(fqn, None)
            if fsdp_param_info is None:
                continue
            state = {} if param_key is None else optim_state_dict[param_key]
            if id(fsdp_param_info) not in all_states:
                all_states[id(fsdp_param_info)] = {}
            all_states[id(fsdp_param_info)][fqn] = state
        elif to_save:
            assert len(optim_state_key.unflat_param_names) == 1
            unflat_param_name = optim_state_key.unflat_param_names[0]
            with SimpleProfiler.profile('none_fsdp_managed_copy'):
                param_key = cast(Union[str, int], param_key)
                fsdp_osd_state[unflat_param_name] = copy.copy(optim_state_dict[param_key])
                if cpu_offload:
                    for state_name, value in sorted_items(fsdp_osd_state[unflat_param_name]):
                        if not torch.is_tensor(value):
                            continue
                        fsdp_osd_state[unflat_param_name][state_name] = value.cpu()
    for _all_states in all_states.values():
        fqn = next(iter(_all_states.keys()))
        fsdp_param_info = fqn_to_fsdp_param_info[fqn]
        assert len(fsdp_param_info.param_requires_grad) > 0, 'With use_orig_params, FSDPParamInfo should have requires_grad information. However, the length is zero.'
        for key, idx in fsdp_param_info.param_indices.items():
            if key in _all_states:
                continue
            if not fsdp_param_info.param_requires_grad[idx]:
                continue
            raise RuntimeError(f'{key} is not in the optimizer state. The FSDPParamInfo has the param keys {sorted(fsdp_param_info.param_indices.keys())} while the optimizer has the param keys {sorted(_all_states.keys())}.')
        fsdp_osd_state.update(_gather_all_orig_param_state(fsdp_param_info, _all_states, shard_state, to_save, cpu_offload))
    return fsdp_osd_state