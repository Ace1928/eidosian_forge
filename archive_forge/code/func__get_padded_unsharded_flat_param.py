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
def _get_padded_unsharded_flat_param(self) -> torch.Tensor:
    """
        Returns a reference to the padded unsharded flat parameter depending on
        the calling context. This should only be called if using a sharded
        strategy.
        """
    self._check_sharded_strategy()
    flat_param = self.flat_param
    if self._force_full_precision and self._uses_param_mixed_precision:
        unsharded_flat_param = flat_param._full_prec_full_param_padded
        _p_assert(unsharded_flat_param.dtype != self._fwd_bwd_param_dtype, f'Expects full precision but got {self._fwd_bwd_param_dtype}')
        if flat_param._full_param_padded.untyped_storage().size() > 0:
            _free_storage(flat_param._full_param_padded)
    else:
        unsharded_flat_param = flat_param._full_param_padded
    return unsharded_flat_param