import itertools
import os
import re
import sys
from abc import ABC, abstractmethod
from contextlib import nullcontext
from copy import deepcopy
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from unittest import mock
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import TrainingState
from torch.distributed.fsdp._init_utils import NO_RESHARD_AFTER_FORWARD_STRATEGIES
from torch.distributed.fsdp.fully_sharded_data_parallel import (
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import always_wrap_policy, ModuleWrapPolicy, wrap
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import FILE_SCHEMA, get_cycles_per_ms
class AlwaysWrapNestedWrappedModule(NestedWrappedModule):

    @staticmethod
    def init(group: dist.ProcessGroup, fsdp_init_mode: FSDPInitMode, cuda_init_mode: CUDAInitMode, fsdp_kwargs: Optional[Dict[str, Any]]=None, deterministic: bool=False):
        """
        Initializes a :class:`NestedWrappedModule` instance, but unlike
        :meth:`NestedWrappedModule.init`, for the ``RECURSIVE`` init mode, this
        wraps with top-level FSDP and the ``always_wrap_policy()`` auto wrap
        policy.
        """
        super_ = super(AlwaysWrapNestedWrappedModule, AlwaysWrapNestedWrappedModule)
        model = super_.init(group=group, fsdp_init_mode=FSDPInitMode.NO_FSDP, cuda_init_mode=cuda_init_mode, fsdp_kwargs=fsdp_kwargs, deterministic=deterministic)
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            return model
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            fsdp_model = FSDP(model, auto_wrap_policy=always_wrap_policy, **fsdp_kwargs)
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            return fsdp_model