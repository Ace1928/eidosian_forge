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
def _root_copy_hook(self, *args: Any, **kwargs: Any) -> None:
    """
        For DDP mixed precision, put low precision copies on separate stream and create events to wait for them.

        When training with DDP mixed precision, this root pre-forward hook kicks
        off low precision copies on a separate stream and creates respective
        events to wait for them.
        """
    self._submodule_to_event = defaultdict(deque)
    with torch.cuda.stream(self._mp_stream):
        for submodule in self.module.modules():
            for param in submodule.parameters(recurse=False):
                if hasattr(param, '_ddp_ignored') and param._ddp_ignored:
                    continue
                _alloc_storage(param._mp_param, param.size())
                with torch.no_grad():
                    param._mp_param.copy_(param.data)
                    if param.grad is not None:
                        param.grad.data = param.grad.to(self.mixed_precision.param_dtype)
                param.data = param._mp_param
            copy_event = torch.cuda.Event()
            copy_event.record()
            self._submodule_to_event[submodule].append(copy_event)