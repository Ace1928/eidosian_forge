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
@contextlib.contextmanager
def _unshard_fsdp_state_params(module: nn.Module, state: _FSDPState, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool):
    """
    This unshards the parameters for a single FSDP state ``state`` that
    corresponds to ``module``.
    """
    _validate_unshard_params_args(state, writeback, rank0_only, offload_to_cpu, with_grads)
    state._device_handle.synchronize()
    maybe_handle = _module_handle(state, module)
    handle = None
    if maybe_handle and maybe_handle._training_state != HandleTrainingState.SUMMON_FULL_PARAMS:
        handle = maybe_handle
    if not handle:
        yield
        return
    assert handle._training_state == HandleTrainingState.IDLE, f'Expects the handle training to be IDLE but got {handle._training_state}'
    handle._training_state = HandleTrainingState.SUMMON_FULL_PARAMS
    _reset_flat_param_grad_info_if_needed(handle)
    free_unsharded_flat_param = handle.needs_unshard()
    computation_stream = state._device_handle.current_stream()
    _unshard(state, handle, computation_stream, computation_stream)
    if with_grads:
        _unshard_grads(handle)
    if rank0_only and state.rank != 0:
        _reshard(state, handle, free_unsharded_flat_param)
        if with_grads:
            _reshard_grads(handle)
        try:
            yield
        finally:
            handle._training_state = HandleTrainingState.IDLE
    else:
        with contextlib.ExitStack() as stack:
            if offload_to_cpu and handle.uses_sharded_strategy:
                stack.enter_context(handle.to_cpu())
            if not state._use_orig_params:
                stack.enter_context(_unflatten_as_params(state, module))
            try:
                yield
            finally:
                stack.close()
                if writeback:
                    _writeback_to_local_shard(handle, with_grads)
                _reshard(state, handle, free_unsharded_flat_param)
                if with_grads:
                    _reshard_grads(handle)
                handle._training_state = HandleTrainingState.IDLE