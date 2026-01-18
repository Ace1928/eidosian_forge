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
def _sharded_post_state_dict_hook(module: nn.Module, fsdp_state: _FSDPState, state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    The hook replaces the unflattened, unsharded parameter in the state_dict
    with a unflattened, sharded parameter (a ShardedTensor).
    """

    def param_hook(state_dict: Dict[str, Any], prefix: str, fqn: str):
        param = state_dict[fqn]
        if not fsdp_state._state_dict_config._use_dtensor:
            sharded_tensor = _ext_chunk_tensor(tensor=param, rank=fsdp_state.rank, world_size=fsdp_state.world_size, num_devices_per_node=fsdp_state._device_handle.device_count(), pg=fsdp_state.process_group, fsdp_extension=fsdp_state._fsdp_extension)
        else:
            sharded_tensor = _ext_chunk_dtensor(tensor=param, rank=fsdp_state.rank, device_mesh=fsdp_state._device_mesh, fsdp_extension=fsdp_state._fsdp_extension)
        if fsdp_state._state_dict_config.offload_to_cpu:
            sharded_tensor = sharded_tensor.cpu()
        state_dict[fqn] = sharded_tensor
    return _common_unshard_post_state_dict_hook(module, fsdp_state, state_dict, prefix, param_hook)