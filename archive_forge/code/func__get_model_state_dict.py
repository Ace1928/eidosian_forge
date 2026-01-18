import contextlib
import functools
import gc
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import (
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint._state_dict_utils import (
from torch.distributed.fsdp import (
from torch.distributed.fsdp._common_utils import (
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel as DDP
def _get_model_state_dict(model: nn.Module, info: _StateDictInfo) -> Dict[str, ValueType]:
    if not info.handle_model:
        return {}
    with info.fsdp_context():
        state_dict = _state_dict_fn(model, 'state_dict')()
    for key in list(state_dict.keys()):
        fqns = _get_fqns(model, key)
        assert len(fqns) == 1
        fqn = next(iter(fqns))
        if fqn != key:

            def verify(key, fqn) -> bool:
                if len(fqn) >= len(key):
                    return False
                fqn_split = fqn.split('.')
                key_split = key.split('.')
                fqn_idx = 0
                for key_idx, key_name in enumerate(key_split):
                    if key_name == fqn_split[fqn_idx]:
                        fqn_idx += 1
                        if fqn_idx == len(fqn_split):
                            return key_idx == len(key_split) - 1
                    elif key_name == 'module':
                        continue
                    else:
                        return False
                return True
            if not verify(key, fqn):
                raise RuntimeError(f'An unexpected key, {key}, exists. FQN is {fqn}')
            state_dict[fqn] = state_dict.pop(key)
    if info.submodule_prefixes:
        new_state_dict: Dict[str, ValueType] = {}
        for fqn in state_dict.keys():
            for prefix in info.submodule_prefixes:
                if not fqn.startswith(prefix):
                    continue
                if info.keep_submodule_prefixes:
                    new_state_dict[fqn] = state_dict[fqn]
                else:
                    new_fqn = fqn[len(prefix):]
                    new_state_dict[new_fqn] = state_dict[fqn]
        state_dict = new_state_dict
    if info.ignore_frozen_params:
        for key, param in model.named_parameters():
            if param.requires_grad:
                continue
            fqns = _get_fqns(model, key)
            for fqn in fqns:
                state_dict.pop(fqn)
    for key, p in list(state_dict.items()):
        if p.is_meta:
            state_dict.pop(key)
    if info.full_state_dict:
        ranks_only = tuple() if not info.cpu_offload else (0,)
        return _gather_state_dict(state_dict, cpu_offload=info.cpu_offload, ranks_only=ranks_only)
    elif info.cpu_offload:
        return _offload_state_dict_to_cpu(state_dict)
    else:
        return state_dict