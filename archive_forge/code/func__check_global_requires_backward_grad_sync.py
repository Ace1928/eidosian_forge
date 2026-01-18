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
def _check_global_requires_backward_grad_sync(self, is_joined_rank):
    if not is_joined_rank and self.require_backward_grad_sync:
        requires_sync_tensor = torch.ones(1, device=self.device)
    else:
        requires_sync_tensor = torch.zeros(1, device=self.device)
    work = dist.all_reduce(requires_sync_tensor, group=self.process_group, async_op=True)
    return work