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
def _setup_output_hook_list(self) -> None:
    """set up a list to avoid registering pre-backward hooks
        incorrectly.
        """
    assert self._is_root, 'This should only be called on the root'
    self._output_pre_backward_hook_registered = []
    for n, m in self.named_modules():
        if n != '' and isinstance(m, FullyShardedDataParallel):
            m._output_pre_backward_hook_registered = self._output_pre_backward_hook_registered