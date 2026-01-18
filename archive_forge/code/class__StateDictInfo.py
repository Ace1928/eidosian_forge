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
@dataclass
class _StateDictInfo(StateDictOptions):
    fqn_param_mapping: Dict[Union[str, torch.Tensor], Union[FQNS_T, torch.Tensor]] = field(default_factory=dict)
    all_fqns: Set[str] = field(default_factory=set)
    submodule_prefixes: Set[str] = field(default_factory=set)
    handle_model: bool = True
    handle_optim: bool = True
    fsdp_context: Callable = contextlib.nullcontext
    fsdp_modules: List[nn.Module] = field(default_factory=list)