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
def _assign_modules_buffers(self):
    """
        Assign self.module.named_buffers to self.modules_buffers.

        Assigns module buffers to self.modules_buffers which are then used to
        broadcast across ranks when broadcast_buffers=True. Note that this
        must be called every time buffers need to be synced because buffers can
        be reassigned by user module,
        see https://github.com/pytorch/pytorch/issues/63916.
        """
    named_module_buffers = [(buffer, buffer_name) for buffer_name, buffer in self.module.named_buffers() if buffer_name not in self.parameters_to_ignore]
    self.modules_buffers = [buffer for buffer, buffer_name in named_module_buffers]
    self.named_module_buffers = {buffer_name: buffer for buffer, buffer_name in named_module_buffers}