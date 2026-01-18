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
@property
def _force_full_precision(self) -> bool:
    return (self._uses_param_mixed_precision or self._uses_reduce_mixed_precision) and (self._training_state == HandleTrainingState.SUMMON_FULL_PARAMS or (not self._fully_sharded_module.training and self._use_full_prec_in_eval))