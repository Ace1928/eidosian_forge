from abc import ABC, abstractmethod
import inspect
from typing import Dict, Type
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.distributed.optim import as_functional_optim
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import (
def _as_overlapped_optim(optim_cls: Type, params, *args, **kwargs):
    """Return a new ``OverlappedOptimizer`` instance that supports ``optim_cls``."""
    for clz in inspect.getmro(optim_cls):
        try:
            return _registered_overlapped_optims[clz](optim_cls, params, *args, **kwargs)
        except KeyError:
            pass
    return _OverlappedStandardOptimizer(optim_cls, params, *args, **kwargs)