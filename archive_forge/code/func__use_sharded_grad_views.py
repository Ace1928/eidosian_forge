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
def _use_sharded_grad_views(self) -> None:
    """
        Sets the original parameter variables' gradients to be flattened
        views into the sharded flat parameter's gradient. This is a no-op if
        there is no gradient.

        Parameters whose data is not present in the sharded flat parameter and
        parameters with ``requires_grad=False`` have their gradients set to
        ``None``. Since the gradient variables do not need to be preserved,
        this method does not manipulate existing ``Tensor`` data directly and
        creates new ``Tensor`` variables instead.
        """
    flat_param = self.flat_param
    self._check_sharded(flat_param)
    grad = self.sharded_grad
    if grad is None:
        for param in chain(flat_param._params, flat_param._shared_params):
            param.grad = None
        return
    self._check_sharded(grad)
    for param, shard_param_info, is_grad_none in zip(flat_param._params, flat_param._shard_param_infos, flat_param._is_grad_none_mask):
        if not shard_param_info.in_shard:
            param.grad = None
        else:
            numel_in_shard = shard_param_info.numel_in_shard
            if param.requires_grad and (not is_grad_none):
                offset = shard_param_info.offset_in_shard
                if self._keep_low_precision_grads or param.dtype != grad.dtype:
                    if param.grad is None:
                        param.grad = torch.empty_like(param)
                    param.grad.data = grad[offset:offset + numel_in_shard].reshape(param.shape)
                else:
                    param.grad = grad[offset:offset + numel_in_shard].reshape(param.shape)
            else:
                param.grad = None
    assert flat_param._shared_params is not None
    for i, (param, (_, _, _, prim_param_name, prim_module, _)) in enumerate(zip(flat_param._shared_params, flat_param._shared_param_infos)):
        in_sharded_flat_param = hasattr(prim_module, prim_param_name)
        if in_sharded_flat_param and param.requires_grad:
            prim_param = getattr(prim_module, prim_param_name)
            param.grad = prim_param.grad
        else:
            param.grad = None