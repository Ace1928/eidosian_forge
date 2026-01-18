from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Union
from functorch import make_fx
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives
import torch.nn as nn
import torch.utils._pytree as pytree
from torch import fx
from torch._decomp.decompositions import native_layer_norm_backward
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._spmd.data_parallel import gradients_tagging
from torch.distributed._spmd.parallel_mode import (
from torch.distributed._tensor import Placement
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo, CodeGen
from torch.nn.utils import stateless
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
def _fused_adam_decomp(self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, *, lr=1, beta1=1, beta2=1, weight_decay=1, eps=1, amsgrad=True, maximize=True, grad_scale=None, found_inf=None):
    orig_tuple = (self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs)
    updated_tuple = aten._fused_adam.default(self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay, eps=eps, amsgrad=amsgrad, maximize=maximize, grad_scale=grad_scale, found_inf=found_inf)
    for idx, (orig, updated) in enumerate(zip(orig_tuple, updated_tuple)):
        if idx == 1:
            continue
        for o, u in zip(orig, updated):
            o.copy_(u)