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
def _free_fp16_param_shard(self, params: Optional[List[Parameter]]=None) -> None:
    """Free storage for FP16 shards for a list of params."""
    if params is None:
        params = self.params
    current_stream = torch.cuda.current_stream()
    for p in params:
        if p._fp16_shard is not None:
            p._fp16_shard.record_stream(current_stream)
            free_storage_(p._fp16_shard)