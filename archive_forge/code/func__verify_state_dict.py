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
def _verify_state_dict(model_state_dict: Dict[str, ValueType], optim_state_dict: OptimizerStateType, info: _StateDictInfo) -> None:
    has_fsdp_root = False
    for module in info.fsdp_modules:
        fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
        assert fsdp_state is not None, 'Expected a fsdp_state with a fsdp module.'
        if fsdp_state._is_root:
            has_fsdp_root = True
            break
    if info.fsdp_modules and (not has_fsdp_root):
        raise RuntimeError('The model has FSDP modules but no FSDP root module exists.')
    if info.handle_model and (not model_state_dict) and (not info.submodule_prefixes) and (not info.ignore_frozen_params) and (not (info.cpu_offload and info.full_state_dict)) and info.strict:
        raise RuntimeError(f'The option indicates that model state_dict is required to save or load, but model state_dict is empty.rank = dist.get_rank()={dist.get_rank()!r}.')
    if info.handle_optim:
        if not (optim_state_dict and optim_state_dict[STATE]) and (not (info.cpu_offload and info.full_state_dict)):
            raise RuntimeError(f'The option indicates that model state_dict is required to save, or load but optim state_dict is empty. {optim_state_dict}')
    for key in model_state_dict.keys():
        if FLAT_PARAM in key:
            raise RuntimeError(f'{key} contains {FLAT_PARAM}. This can happen if the model is not the root module.')