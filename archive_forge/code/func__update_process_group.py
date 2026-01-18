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
def _update_process_group(self, new_process_group):
    """
        Dynamically updates the process group for DDP so that we can shrink/expand DDP
        world size without having to reinitialize DDP.

        NOTE: If you are using custom communications hooks via, register_comm_hook,
        you need to update the process groups for those hooks separately.
        """
    self._has_rebuilt_buckets = False
    self.reducer._reset_state()
    if not _rank_not_in_group(new_process_group):
        self.process_group = new_process_group
        self.reducer._update_process_group(new_process_group)