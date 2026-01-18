import contextlib
import warnings
from typing import cast, Generator
import torch
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.utils import _p_assert
from ._flat_param import FlatParamHandle
def _validate_unshard_params_args(state: _FSDPState, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool) -> None:
    if with_grads and (offload_to_cpu or not state._use_orig_params):
        raise NotImplementedError(f'with_grads={with_grads}, use_orig_params={state._use_orig_params}, offload_to_cpu={offload_to_cpu} is not supported yet')
    if offload_to_cpu and state._handle and (not state._handle.uses_sharded_strategy):
        raise NotImplementedError('offload_to_cpu=True and NO_SHARD is not supported yet')
    if writeback and rank0_only:
        raise NotImplementedError('writeback=True and rank0_only=True is not supported yet')
    if offload_to_cpu and (not rank0_only):
        warnings.warn('offload_to_cpu=True and rank0_only=False may result in theunsharded parameters being redundantly copied to CPU memory for GPUs sharing the same CPU memory, which risks CPU OOM. We recommend using offload_to_cpu=True with rank0_only=True.')