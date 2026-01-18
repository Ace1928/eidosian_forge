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
def _load_optim_state_dict(model: nn.Module, optimizers: Tuple[torch.optim.Optimizer, ...], state_dict: OptimizerStateType, info: _StateDictInfo) -> None:
    if not info.handle_optim:
        return
    for optim in optimizers:
        optim_state_dict = _split_optim_state_dict(model, optim, state_dict, info)
        if info.fsdp_modules:
            with info.fsdp_context():
                optim_state_dict = FSDP.optim_state_dict_to_load(model, optim, optim_state_dict)
        _init_optim_state(optim)
        _state_dict_fn(optim, 'load_state_dict')(state_dict=optim_state_dict)