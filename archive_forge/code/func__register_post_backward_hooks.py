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
def _register_post_backward_hooks(self) -> None:
    """
        Register backward hooks to reshard params and reduce-scatter grads.

        This is called during forward pass. The goal is to attach a hook
        on each of the parameter's gradient generating function (``grad_acc``
        below) so that the hook is called *after* all gradients for that
        param are computed.

        Goals:

        1. We want the hook to fire once and only once *after* all gradients
        are accumulated for a param.
        2. If it fires more than once, we end up incorrectly shard the grad
        multiple times. (could lead to dimension too small)
        3. If it fires once but too early or doesn't fire, we leave gradients
        unsharded. (could lead to dimension too large)

        There are several cases here:
        1. We can call the same module multiple times in a single outer forward
           pass. We register multiple hooks but autograd should fire the last
           one after the total gradient is computed and accumulated. If it does
           fire multiple times, we may have a crash due to gradient being already
           sharded and shape mismatch.
           On the other hand, due to _saved_grad_shard, this case may also work
           but with extra grad scatter-gather.
        2. With activation checkpointing and case 1.
        3. The same outer forward can be called multiple times before any backward
           is called (within the no_sync context) for a special way of gradient
           accumulation. (see test_fsdp_fwd_fwd_bwd_bwd.py)
        4. When a param is shared by multiple FSDP wrapper instances, this can
           register multiple times. (See test_fsdp_shared_weights.py)

        It appears that registering the hook everytime and let them fire and
        hook being removed/freed automatically is the correct thing to do. But this
        is purely based on experiments.
        """
    if not torch.is_grad_enabled():
        return
    for p in self.params:
        if p.requires_grad:
            p_tmp = p.expand_as(p)
            assert p_tmp.grad_fn is not None
            grad_acc = p_tmp.grad_fn.next_functions[0][0]
            handle = grad_acc.register_hook(functools.partial(self._post_backward_hook, p))
            p._shard_bwd_hook = (grad_acc, handle)