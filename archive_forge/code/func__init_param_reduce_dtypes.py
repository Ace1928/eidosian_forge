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
def _init_param_reduce_dtypes(self, mp_param_dtype: Optional[torch.dtype], mp_reduce_dtype: Optional[torch.dtype]) -> None:
    """
        Precondition: ``self.flat_param`` is set. This ensures that this
        handle's parameters have a single dtype.

        Postcondition: This sets ``self._fwd_bwd_param_dtype`` and
        ``self._reduce_dtype``. If ``mp_param_dtype`` or ``mp_reduce_dtype``
        is ``None``, then we assume the original parameter dtype. One special
        case is if ``mp_param_dtype`` is not ``None`` and ``mp_reduce_dtype``
        is ``None``, in which case we assume the gradient reduction dtype
        matches the forward/backward parameter dtype.
        """
    self._low_prec_param_dtype_specified = mp_param_dtype is not None
    self._low_prec_reduce_dtype_specified = mp_reduce_dtype is not None
    if self._low_prec_param_dtype_specified and (not self._low_prec_reduce_dtype_specified):
        self._fwd_bwd_param_dtype = mp_param_dtype
        self._reduce_dtype = self._fwd_bwd_param_dtype
    else:
        self._fwd_bwd_param_dtype = mp_param_dtype or self._orig_param_dtype
        self._reduce_dtype = mp_reduce_dtype or self._orig_param_dtype
    assert self._fwd_bwd_param_dtype is not None
    assert self._reduce_dtype is not None