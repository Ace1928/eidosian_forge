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
def _common_unshard_post_state_dict_hook(module: nn.Module, fsdp_state: _FSDPState, state_dict: Dict[str, Any], prefix: str, param_hook: Callable) -> Dict[str, Any]:
    """
    The post-state_dict flow that shared by all state_dict types that require
    ``_unshard_fsdp_state_params()``. FULL_STATE_DICT and SHARDED_STATE_DICT use this
    hook.
    """
    _replace_by_prefix(state_dict, prefix + f'{FSDP_PREFIX}', prefix)
    if not state_dict or not _has_fsdp_params(fsdp_state, module):
        if not (_is_composable(fsdp_state) and fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD):
            _exit_unshard_params_ctx(module, fsdp_state)
        return state_dict
    rank0_only = fsdp_state._state_dict_type == StateDictType.FULL_STATE_DICT and cast(FullStateDictConfig, fsdp_state._state_dict_config).rank0_only
    no_fsdp_return = rank0_only and fsdp_state.rank != 0
    if no_fsdp_return and (not fsdp_state._use_orig_params):
        for clean_key in fsdp_state._buffer_names:
            clean_key = clean_key.replace(f'{checkpoint_wrapper._CHECKPOINT_PREFIX}.', '')
            state_dict.pop(f'{prefix}{clean_key}', None)
        state_dict.pop(f'{prefix}{FLAT_PARAM}')
        _exit_unshard_params_ctx(module, fsdp_state)
        return state_dict
    for fqn, param_name, module_name in _param_name_infos(module, fsdp_state):
        fqn = f'{prefix}{fqn}'
        if no_fsdp_return:
            state_dict.pop(fqn)
            continue
        assert fqn in state_dict, f'FSDP assumes {fqn} is in the state_dict but the state_dict only has {state_dict.keys()}. prefix={prefix}, module_name={module_name}, param_name={param_name} rank={fsdp_state.rank}.'
        param_hook(state_dict, prefix, fqn)
    if not (_is_composable(fsdp_state) and fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD):
        _exit_unshard_params_ctx(module, fsdp_state)
    cpu_device = torch.device('cpu')
    buffer_clean_fqns = []
    buffers = []
    for clean_key in fsdp_state._buffer_names:
        clean_key = clean_tensor_name(clean_key)
        fqn = f'{prefix}{clean_key}'
        if fqn not in state_dict:
            continue
        if no_fsdp_return:
            state_dict.pop(fqn)
        else:
            buffer = state_dict[fqn]
            if fsdp_state._state_dict_config.offload_to_cpu and buffer.device != cpu_device:
                state_dict[fqn] = buffer.to(cpu_device)
            if clean_key not in fsdp_state._ignored_buffer_names:
                buffer_clean_fqns.append(clean_key)
                buffers.append(state_dict[fqn])
    if buffers:
        mixed_precision_enabled_for_buffers = fsdp_state._mixed_precision_enabled_for_buffers() if not _is_composable(fsdp_state) else fsdp_state.mixed_precision.buffer_dtype is not None
        if mixed_precision_enabled_for_buffers:
            buffer_dtypes = _get_orig_buffer_dtypes(fsdp_state, buffer_clean_fqns)
            _cast_buffers_to_dtype_and_device(buffers, buffer_dtypes, fsdp_state.compute_device)
            for buffer, clean_fqn in zip(buffers, buffer_clean_fqns):
                fqn = f'{prefix}{clean_fqn}'
                logger.info('FSDP is casting the dtype of %s to %s', fqn, buffer.dtype)
                state_dict[fqn] = buffer.clone()
    return state_dict