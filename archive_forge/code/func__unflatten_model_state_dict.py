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
def _unflatten_model_state_dict(model: nn.Module, state_dict: Union[Dict[nn.Module, Dict[str, ValueType]], Dict[str, ValueType]]) -> Dict[str, ValueType]:
    if not state_dict:
        return {}
    if isinstance(next(iter(state_dict.keys())), nn.Module):
        cast_state_dict = cast(Dict[nn.Module, Dict[str, ValueType]], state_dict)
        new_state_dict: Dict[str, ValueType] = {}
        for submodule, sub_state_dict in cast_state_dict.items():
            for name, m in model.named_modules():
                if m != submodule:
                    continue
                fqns = _get_fqns(model, name)
                assert len(fqns) == 1, 'FQNs for a submodule should only have 1 element'
                prefix = f'{next(iter(fqns))}.'
                new_state_dict.update({prefix + subfqn: value for subfqn, value in sub_state_dict.items()})
        return new_state_dict
    else:
        return cast(Dict[str, ValueType], state_dict)