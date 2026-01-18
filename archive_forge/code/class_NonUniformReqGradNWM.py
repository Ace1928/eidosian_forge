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
class NonUniformReqGradNWM(NestedWrappedModule):

    def __init__(self, group: dist.ProcessGroup, wrap_fsdp: bool, cuda_init_mode: CUDAInitMode, deterministic: bool, **fsdp_kwargs):
        super(NestedWrappedModule, self).__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        move_to_cuda = cuda_init_mode == CUDAInitMode.CUDA_BEFORE

        def _maybe_wrap(layer):
            if wrap_fsdp:
                return FSDP(layer, group, **fsdp_kwargs)
            return layer
        if deterministic:
            torch.manual_seed(0)
        self.module = nn.Sequential(_maybe_cuda(nn.Linear(8, 4), move_to_cuda), _maybe_wrap(nn.Sequential(_maybe_wrap(_maybe_cuda(nn.Linear(4, 16), move_to_cuda)), _maybe_cuda(nn.Linear(16, 16), move_to_cuda))), _maybe_wrap(nn.Sequential(_maybe_cuda(nn.Linear(16, 4), move_to_cuda), _maybe_cuda(nn.Linear(4, 8), move_to_cuda))))

    @staticmethod
    def _set_nonuniform_req_grad(model, req_grad_mask) -> None:
        for n, p in model.named_parameters():
            if not re.match(req_grad_mask, n):
                p.requires_grad_(False)

    @staticmethod
    def init(group: dist.ProcessGroup, fsdp_init_mode: FSDPInitMode, cuda_init_mode: CUDAInitMode, fsdp_kwargs: Optional[Dict[str, Any]]=None, deterministic: bool=False):
        """
        Initializes a :class:`NestedWrappedModule` instance, but unlike
        :meth:`NestedWrappedModule.init`, it wraps a second :class:`torch.nn.Sequential`
        container to enable the desired non-uniform ``requires_grad``
        ``use_orig_params=True`` tests. For both ``RECURSIVE`` and ``NO_FSDP``
        init modes, freezes all parameters except the last two to validate
        ``ShardedGradScaler`` support for ranks with no (non-zero sized) local shards in
        FSDP ``use_orig_params=True`` mode.
        """
        req_grad_pattern = re.compile('module\\.2.*\\.1.*')
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            ddp_model = NonUniformReqGradNWM(group, wrap_fsdp=False, cuda_init_mode=cuda_init_mode, deterministic=deterministic)
            NonUniformReqGradNWM._set_nonuniform_req_grad(ddp_model, req_grad_pattern)
            return ddp_model
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            if fsdp_kwargs is None:
                fsdp_kwargs = {}
            fsdp_model = NonUniformReqGradNWM(group, wrap_fsdp=True, cuda_init_mode=cuda_init_mode, deterministic=deterministic, **fsdp_kwargs)
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            NonUniformReqGradNWM._set_nonuniform_req_grad(fsdp_model, req_grad_pattern)
            return fsdp_model
        raise ValueError(f'Unsupported FSDP init mode: {fsdp_init_mode}')