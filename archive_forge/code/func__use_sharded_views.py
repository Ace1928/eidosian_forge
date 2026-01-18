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
def _use_sharded_views(self) -> None:
    """
        Sets the original parameter variables' data to be flattened views into
        the sharded flat parameter.

        The views are kept as flattened to simplify the case where a parameter
        is sharded across ranks. Parameters whose data is not present in the
        sharded flat parameter have their data set to a size-0 empty tensor. We
        do not delete them to ensure to preserve expected behaviors like model
        printability. Parameters whose data is present must preserve their
        variables to be passable to an optimizer.
        """
    self._unsharded_flat_param_for_skipped_views = None
    if not self.uses_sharded_strategy:
        self._use_unsharded_views(as_params=True)
        return
    flat_param = self.flat_param
    self._check_sharded(flat_param)
    size_0_empty_tensor = torch.empty(0, dtype=self.flat_param.dtype, device=self.flat_param.device, requires_grad=False)
    for param, shard_param_info, (param_name, module, _) in zip(flat_param._params, flat_param._shard_param_infos, flat_param._param_infos):
        self._setattr_param(module, param_name, param)
        if not shard_param_info.in_shard:
            param.data = size_0_empty_tensor
        else:
            offset = shard_param_info.offset_in_shard
            numel_in_shard = shard_param_info.numel_in_shard
            param.data = flat_param[offset:offset + numel_in_shard]
    assert self.flat_param._shared_params is not None
    for i, (param, (param_name, module, _, prim_param_name, prim_module, _)) in enumerate(zip(self.flat_param._shared_params, self.flat_param._shared_param_infos)):
        self._setattr_param(module, param_name, param)
        prim_param = getattr(prim_module, prim_param_name)
        param.data = prim_param
    if self._training_state == HandleTrainingState.BACKWARD_POST:
        for i in range(len(self.flat_param._tensors)):
            self.flat_param._tensors[i] = None