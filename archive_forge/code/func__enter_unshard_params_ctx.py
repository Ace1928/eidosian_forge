import contextlib
import logging
import math
import warnings
from typing import Any, Callable, cast, Dict, Generator, Iterator, no_type_check, Tuple
import torch
import torch.distributed as dist
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as checkpoint_wrapper
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import _mesh_resources
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._debug_utils import SimpleProfiler
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.fsdp.api import (
from torch.distributed.utils import _replace_by_prefix
from ._fsdp_extensions import (
from ._unshard_param_utils import _unshard_fsdp_state_params, FLAT_PARAM
@no_type_check
def _enter_unshard_params_ctx(module: nn.Module, fsdp_state: _FSDPState, writeback: bool=False, rank0_only: bool=False, offload_to_cpu: bool=False, with_grads: bool=False) -> None:
    """
    state_dict hooks cannot use the pure context call as the checkpoint flow
    requires to enter the context in the pre-hook but leave the context in the
    post-hook. This API enters the context of ``_unshard_fsdp_state_params``.
    """
    assert module not in fsdp_state._unshard_params_ctx, 'Entering the ``_unshard_fsdp_state_params`` context but _unshard_params_ctx[module] is not None.'
    fsdp_state._unshard_params_ctx[module] = _unshard_fsdp_state_params(module, fsdp_state, writeback=writeback, rank0_only=rank0_only, offload_to_cpu=offload_to_cpu, with_grads=with_grads)
    fsdp_state._unshard_params_ctx[module].__enter__()