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
def _local_post_state_dict_hook(module: nn.Module, fsdp_state: _FSDPState, state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    This hook create a ShardedTensor from the local flat_param and replace
    the state_dict[f"{prefix}{FLAT_PARAM}] with the ShardedTensor. No copy
    will happen. The underlying storage is the same.
    """
    _replace_by_prefix(state_dict, f'{prefix}{FSDP_PREFIX}', prefix)
    if not _has_fsdp_params(fsdp_state, module):
        return state_dict
    assert _module_handle(fsdp_state, module), 'Should have returned early'
    flat_param = _module_handle(fsdp_state, module).flat_param
    full_numel = flat_param._unpadded_unsharded_size.numel()
    shard_offset = flat_param.numel() * fsdp_state.rank
    valid_data_size = flat_param.numel() - flat_param._shard_numel_padded
    if valid_data_size > 0:
        flat_param = flat_param[:valid_data_size].view(valid_data_size)
        local_shards = [Shard.from_tensor_and_offsets(flat_param, [shard_offset], fsdp_state.rank)]
    else:
        local_shards = []
    sharded_tensor = init_from_local_shards(local_shards, full_numel, process_group=fsdp_state.process_group)
    if fsdp_state._state_dict_config.offload_to_cpu:
        sharded_tensor = sharded_tensor.cpu()
    state_dict[f'{prefix}{FLAT_PARAM}'] = sharded_tensor
    return state_dict