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
def _set_is_root(self) -> None:
    """If ``True``, implies that no other :class:`FullyShardedDataParallel`
        instance wraps this one. Called once by :func:`_lazy_init`.
        Also sets self.children_share_process_group = True if all child
        instances share the same process group. If some child instances use a
        different process group, self.clip_grad_norm_ will raise an error.
        """
    if self._is_root is not None:
        return
    self._is_root = True
    self.assert_state(TrainingState.IDLE)
    self.children_share_process_group = True
    for n, m in self.named_modules():
        if n != '' and isinstance(m, FullyShardedDataParallel):
            m._is_root = False
            if m.process_group != self.process_group:
                self.children_share_process_group = False
            m.no_broadcast_optim_state = m.no_broadcast_optim_state or (m.world_size == 1 and m.world_size < self.world_size and (m.process_group != self.process_group))