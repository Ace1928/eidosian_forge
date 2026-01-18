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
def _split_optim_state_dict(model: nn.Module, optim: torch.optim.Optimizer, optim_state_dict: OptimizerStateType, info: _StateDictInfo) -> OptimizerStateType:
    """
    Extract the corresponding optim state_dict from ``optim_state_dict`` for
    ``optim`` and return the result optim state_dict.

    Args:
        model (nn.Module): the root model.
        optim (torch.optim.Optimizer): the optimizer.
        optim_state_dict (Dict[str, ValueType]): the superset optim state_dict that
            contains the optim state_dict of ``optim``.
        info (_StateDictInfo): state dict information.

    Returns:
        The optim state_dict of ``optim``.
    """
    state: DictValueType = {}
    pg_state: ListDictValueType = []
    return_osd: OptimizerStateType = {STATE: state, PG: pg_state}
    pg_mapping: Dict[int, int] = {}
    for param_group in optim.param_groups:
        pg_state.append({PARAMS: []})
        for param in param_group[PARAMS]:
            for fqn in info.fqn_param_mapping[param]:
                params = pg_state[-1][PARAMS]
                assert isinstance(params, list)
                params.append(fqn)
                if param.requires_grad:
                    state[fqn] = cast(DictValueType, optim_state_dict[STATE])[fqn]
                for loaded_param_group in cast(ListDictValueType, optim_state_dict[PG]):
                    params = loaded_param_group[PARAMS]
                    assert isinstance(params, list)
                    if fqn in params:
                        pg_mapping[id(loaded_param_group)] = len(return_osd[PG]) - 1
    for param_group in cast(ListDictValueType, optim_state_dict[PG]):
        idx = pg_mapping.get(id(param_group), -1)
        if idx == -1:
            continue
        for key, value in param_group.items():
            if key == PARAMS:
                continue
            pg_state[idx][key] = value
    return return_osd