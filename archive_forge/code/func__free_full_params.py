import contextlib
import copy
from enum import Enum, auto
import functools
import logging
from math import inf
import os
import time
import traceback
import typing
from typing import (
import torch
from torch.autograd import Variable
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from fairscale.internal.containers import apply_to_tensors
from fairscale.internal.parallel import (
from fairscale.internal.params import calc_grad_norm, recursive_copy_to_device
from fairscale.internal.reduce_scatter_bucketer import ReduceScatterBucketer
from fairscale.internal.state_dict import replace_by_prefix_
from fairscale.nn.misc import FlattenParamsWrapper, _enable_pre_load_state_dict_hook
from fairscale.nn.wrap import auto_wrap, config_auto_wrap_policy, enable_wrap
from . import fsdp_optim_utils as ou
@torch.no_grad()
def _free_full_params(self, params: Optional[List[Parameter]]=None) -> None:
    """Free up storage for full parameters."""
    if params is None:
        params = self.params
    self.has_full_params = False
    current_stream = torch.cuda.current_stream()
    for p in params:
        if not p._is_sharded:
            if self.mixed_precision or self.move_params_to_cpu:
                self._free_fp16_param_shard([p])
            continue
        p._full_param_padded.record_stream(current_stream)
        free_storage_(p._full_param_padded)