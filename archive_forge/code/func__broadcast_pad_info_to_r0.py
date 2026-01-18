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
def _broadcast_pad_info_to_r0(self) -> List[List[List[int]]]:
    """Collect [x.numel_padded_per_param for x in get_fsdp_instances(self)] from each rank."""
    world_pad_info: List[List[List[int]]] = []
    my_pad_info: List[List[int]] = [cast(List[int], m.numel_padded_per_param) for m in get_fsdp_instances(self, skip_empty=True)]
    for rank in range(self.world_size):
        if rank == self.rank:
            pad_info = my_pad_info
        else:
            pad_info = [[0]] * len(my_pad_info)
        dist.broadcast_object_list(pad_info, src=rank, group=self.process_group)
        if self.rank == 0:
            world_pad_info.append(pad_info)
    return world_pad_info