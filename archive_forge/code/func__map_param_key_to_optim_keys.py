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
def _map_param_key_to_optim_keys(optim_state_dict: Dict[str, Any], group: Optional[dist.ProcessGroup], param_key_to_param: Dict[Union[int, str], nn.Parameter], param_to_fqns: Dict[nn.Parameter, List[str]], fqn_to_fsdp_param_info: Dict[str, FSDPParamInfo], merge_keys: bool=False) -> Tuple[List[_OptimStateKey], Dict[_OptimStateKey, Union[int, str]]]:
    """
    Construct the local mapping between the ``_OptimStateKey`` and parameter keys
    and all the ``_OptimStateKey`` across ranks. If ``merge_keys`` is False, rank0
    must contain all the ``_OptimStateKey``, an exception will be raised otherwise.
    Note that ``merge_keys`` should equal to ``use_orig_params``.
    """
    rank = dist.get_rank(group)
    optim_state_key_to_param_key: Dict[_OptimStateKey, Union[int, str]] = {}
    all_optim_state_keys: List[_OptimStateKey] = []
    for param_key, param in param_key_to_param.items():
        if param_key not in optim_state_dict['state']:
            continue
        fqns = param_to_fqns[param]
        is_fsdp_managed = isinstance(param, FlatParameter)
        if is_fsdp_managed:
            assert fqns[0] in fqn_to_fsdp_param_info, (fqns[0], list(fqn_to_fsdp_param_info.keys()))
        is_fsdp_managed = fqns[0] in fqn_to_fsdp_param_info
        optim_state_key = _OptimStateKey(unflat_param_names=tuple(fqns), is_fsdp_managed=is_fsdp_managed)
        if rank == 0 or merge_keys:
            all_optim_state_keys.append(optim_state_key)
        optim_state_key_to_param_key[optim_state_key] = param_key
    if merge_keys:
        all_keys: List[List[_OptimStateKey]] = [[] for _ in range(dist.get_world_size(group))]
        dist.all_gather_object(all_keys, all_optim_state_keys, group=group)
        merge_all_optim_state_keys = [key for local_keys in all_keys for key in local_keys]
        all_optim_state_keys = sorted(set(merge_all_optim_state_keys))
    else:
        key_obj_list: List[Optional[List[_OptimStateKey]]] = [all_optim_state_keys] if rank == 0 else [None]
        dist.broadcast_object_list(key_obj_list, src=0, group=group)
        assert key_obj_list[0] is not None
        all_optim_state_keys = key_obj_list[0]
        _check_missing_keys_on_rank(all_optim_state_keys, optim_state_key_to_param_key, param_key_to_param, group)
    return (all_optim_state_keys, optim_state_key_to_param_key)