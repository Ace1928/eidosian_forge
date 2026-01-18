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
def _full_pre_state_dict_hook(fsdp_state: _FSDPState, module: nn.Module, *args, **kwargs) -> None:
    """
    Hook that runs before model.state_dict() is called. pre-state_dict hook is
    not actually supported by ``nn.Module``. As a result, this API is called
    from ``_full_post_state_dict_hook()`` to simulate the case. Once pre-state_dict
    is supported in ``nn.Module``, this hook will be registered as a hook in
    ``nn.Module``.
    """
    if getattr(fsdp_state, '_device_mesh', False):
        parent_mesh = _mesh_resources.get_parent_mesh(fsdp_state._device_mesh)
        if parent_mesh:
            raise RuntimeError(f"Found FSDP's device_mesh {fsdp_state._device_mesh} has a parent device_mesh {parent_mesh}.", 'We do not support FULL_STATE_DICT for 2D FSDP + TP. Please use FSDP SHARDED_STATE_DICT instead.')
    _common_pre_state_dict_hook(module, fsdp_state)
    _common_unshard_pre_state_dict_hook(module, fsdp_state, offload_to_cpu=fsdp_state._state_dict_config.offload_to_cpu, rank0_only=cast(FullStateDictConfig, fsdp_state._state_dict_config).rank0_only)