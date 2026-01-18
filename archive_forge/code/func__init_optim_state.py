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
def _init_optim_state(optim: torch.optim.Optimizer) -> None:
    """
    Initialize optim states by calling the step() with zero grads.
    """
    if optim.state:
        return
    for param_group in optim.param_groups:
        for param in param_group[PARAMS]:
            if param.grad is not None:
                raise RuntimeError('state_dict can only be used if the optimizer states are initialized (usually after one step() with gradients) or gradients are None. For the later case, state_dict will fake the gradients as zero to initialize the optimizer states. However, the gradients are not None.')
            if param.requires_grad:
                param.grad = torch.zeros_like(param)
    optim.step(closure=None)
    optim.zero_grad(set_to_none=True)