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
def _setup_in_backward_optimizers(self):
    if any((hasattr(p, '_in_backward_optimizers') for p in self._module_parameters)):
        torch._C._log_api_usage_once('ddp.optimizer_in_backward')
        param_to_handle_map = dist.optim.apply_optimizer_in_backward.param_to_optim_hook_handle_map
        for p in self._module_parameters:
            for handle in param_to_handle_map.get(p, []):
                handle.remove()
        ddp_weakref = weakref.ref(self)
        from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import _apply_optim_in_backward_hook
        self.register_comm_hook(ddp_weakref, _apply_optim_in_backward_hook(gradient_is_bucket_view=self.gradient_as_bucket_view))
        self.reducer._set_optimizer_in_backward()