import copy
import functools
import inspect
import itertools
import logging
import os
import sys
import warnings
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import auto, Enum
from typing import Any, Callable, List, Optional, Type
import torch
import torch.distributed as dist
from torch.autograd import Function, Variable
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch._utils import _get_device_index
from ..modules import Module
from .scatter_gather import gather, scatter_kwargs  # noqa: F401
def _register_delay_all_reduce_hook(self, bucket_cap_mb, param_to_hook_all_reduce, device_ids):
    device = torch.device('cpu') if device_ids is None else device_ids[0]
    self._delay_grad_buffer = torch.zeros(sum([p.numel() for p in self._delay_all_reduce_params]), device=device)
    detached_params = [p.detach() for p in self._delay_all_reduce_params]
    dist._broadcast_coalesced(self.process_group, detached_params, bucket_cap_mb, 0)
    param_to_hook_all_reduce.register_hook(self._delayed_all_reduce_hook)
    offset = 0
    for param in self._delay_all_reduce_params:
        grad_view = self._delay_grad_buffer[offset:offset + param.numel()].view(param.shape)
        self._delay_grad_views.append(grad_view)
        offset = offset + param.numel()
    for module_name, module in self.module.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                full_name = f'{module_name}.{param_name}'
                if full_name not in self.parameters_to_ignore:
                    return
    self._delay_all_reduce_all_params = True