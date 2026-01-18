import contextlib
import functools
import logging
import os
import warnings
from enum import auto, Enum
from itertools import accumulate, chain
from typing import (
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.fsdp._common_utils import (
from torch.distributed.utils import _alloc_storage, _free_storage, _p_assert
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]
from ._fsdp_extensions import _ext_post_unflatten_transform, _ext_pre_flatten_transform
@no_type_check
@torch.no_grad()
def _writeback_orig_params(self) -> bool:
    """
        Iterates over the original parameters and writes back any parameters
        that changed storages (due to a non-inplace operator) to the handle's
        ``FlatParameter``. This method preserves the ``FlatParameter` 's
        device even if an original parameter's device changes.

        Raises:
            RuntimeError: If an original parameter or gradient changes storages
            but no longer has the expected flattened shape.
        Returns: ``True`` if some writeback happened, and ``False`` otherwise.
        """
    if self.uses_sharded_strategy and (not self.is_sharded(self.flat_param)) and (not self._skipped_use_sharded_views):
        return False
    flat_param = self.flat_param
    wroteback = False
    if self._skipped_use_sharded_views and self.uses_sharded_strategy:
        flat_param_data_ptr = self._unsharded_flat_param_for_skipped_views.untyped_storage().data_ptr()
        _p_assert(flat_param_data_ptr > 0, 'If skipped using sharded views, the unsharded flat parameter should be allocated')
    else:
        flat_param_data_ptr = flat_param.untyped_storage().data_ptr()
    flat_param_grad = flat_param.grad if self.uses_sharded_strategy or not self._offload_params else flat_param._cpu_grad
    flat_param_grad_data_ptr = None if flat_param_grad is None else flat_param_grad.untyped_storage().data_ptr()
    for i, (param, (in_shard, offset_in_shard, numel_in_shard, _, _), (param_name, module, _)) in enumerate(zip(flat_param._params, flat_param._shard_param_infos, flat_param._param_infos)):
        if not in_shard:
            continue
        if not hasattr(module, param_name):
            continue
        if self._skipped_use_sharded_views:
            param = flat_param._tensors[i]
            _p_assert(param is not None, f'Expects to have saved tensor for {flat_param._fqns[i]}')
        param_changed = getattr(module, param_name) is not param
        needs_param_writeback = param_changed or not _same_storage_as_data_ptr(param, flat_param_data_ptr)
        if self._skipped_use_sharded_views and (param_changed or needs_param_writeback):
            raise AssertionError(f'FSDP does not support changing the parameters between forward and backward for {self._sharding_strategy}')
        if param_changed:
            param = getattr(module, param_name)
            flat_param._params[i] = param
        if needs_param_writeback:
            expected_shape = torch.Size([numel_in_shard])
            self._writeback_tensor(param, flat_param, i, expected_shape, offset_in_shard, True)
            wroteback = True
        if self._skipped_use_sharded_views:
            continue
        if param.grad is None and flat_param.grad is not None:
            expected_shape = torch.Size([numel_in_shard])
            self._writeback_tensor(None, flat_param.grad, i, expected_shape, offset_in_shard, False)
        elif param.grad is not None:
            if not self.uses_sharded_strategy and self._offload_params:
                continue
            needs_grad_writeback = flat_param_grad is None or not _same_storage_as_data_ptr(param.grad, flat_param_grad_data_ptr)
            if needs_grad_writeback:
                if flat_param_grad is None:
                    flat_param_grad = torch.zeros_like(flat_param)
                expected_shape = torch.Size([numel_in_shard])
                self._writeback_tensor(param.grad, flat_param_grad, i, expected_shape, offset_in_shard, False)
                flat_param.grad = flat_param_grad
                flat_param_grad = flat_param.grad
                flat_param_grad_data_ptr = flat_param_grad.untyped_storage().data_ptr()
    for i, (param_name, module, _, prim_param_name, prim_module, _) in enumerate(flat_param._shared_param_infos):
        if getattr(module, param_name) is not getattr(prim_module, prim_param_name):
            raise NotImplementedError('Changing shared parameters is not supported yet')
    return wroteback