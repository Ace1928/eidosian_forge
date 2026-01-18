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
def _reset_is_grad_none(self) -> None:
    """
        Resets ``_is_grad_none_mask`` as needed. This method should only be
        called in the post-backward after gradient computation, in which case
        if a parameter requires gradient, then it will surely receive a
        gradient and we may reset its mask entry to ``False``.
        """
    if not self._use_orig_params:
        return
    _p_assert(self._training_state == HandleTrainingState.BACKWARD_POST, 'Expects to only be called in the post-backward after gradient computation')
    flat_param = self.flat_param
    assert flat_param._params is not None
    for i, param in enumerate(flat_param._params):
        if param.requires_grad:
            assert flat_param._is_grad_none_mask is not None
            flat_param._is_grad_none_mask[i] = False