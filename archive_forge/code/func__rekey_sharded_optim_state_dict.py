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
def _rekey_sharded_optim_state_dict(sharded_osd: Dict[str, Any], model: nn.Module, optim: torch.optim.Optimizer, optim_input: Optional[Union[List[Dict[str, Any]], Iterable[nn.Parameter]]], using_optim_input: bool, is_named_optimizer: bool=False) -> Dict[str, Any]:
    """
    Rekeys the optimizer state dict from unflattened parameter names to flat
    parameter IDs according to the calling rank's ``optim``, which may be
    different across ranks. In particular, the unflattened parameter names are
    represented as :class:`_OptimStateKey` s.
    """
    param_to_fqns = _get_param_to_fqns(model)
    flat_param_to_fqn = _get_flat_param_to_fqn(model)
    param_to_param_key: Dict[nn.Parameter, Union[int, str]] = cast(Dict[nn.Parameter, Union[int, str]], _get_param_to_param_id_from_optim_input(model, optim_input) if using_optim_input else _get_param_to_param_key(optim, model, is_named_optimizer, param_to_fqns, flat_param_to_fqn))
    assert len(param_to_param_key) <= len(param_to_fqns)
    unflat_param_names_to_flat_param_key: Dict[Tuple[str, ...], Union[int, str]] = {}
    unflat_param_name_to_flat_param_key: Dict[str, Union[int, str]] = {}
    for param, unflat_param_names in param_to_fqns.items():
        if param not in param_to_param_key:
            continue
        flat_param_key = param_to_param_key[param]
        unflat_param_names_to_flat_param_key[tuple(unflat_param_names)] = flat_param_key
        for unflat_param_name in unflat_param_names:
            unflat_param_name_to_flat_param_key[unflat_param_name] = flat_param_key
    sharded_osd_state = sharded_osd['state']
    rekeyed_osd_state: Dict[Union[str, int], Any] = {}
    for key, param_state in sharded_osd_state.items():
        if isinstance(key, str):
            rekeyed_osd_state[key] = param_state
            continue
        flat_param_key = unflat_param_names_to_flat_param_key.get(key.unflat_param_names, key.unflat_param_names)
        rekeyed_osd_state[flat_param_key] = param_state
    if 'param_groups' in sharded_osd:
        rekeyed_osd_param_groups: List[Dict[str, Any]] = []
        for unflat_param_group in sharded_osd['param_groups']:
            flat_param_group = copy.deepcopy(unflat_param_group)
            flat_param_keys = sorted({unflat_param_name_to_flat_param_key[unflat_param_name] for unflat_param_name in unflat_param_group['params']})
            flat_param_group['params'] = flat_param_keys
            rekeyed_osd_param_groups.append(flat_param_group)
        return {'state': rekeyed_osd_state, 'param_groups': rekeyed_osd_param_groups}
    else:
        return {'state': rekeyed_osd_state}