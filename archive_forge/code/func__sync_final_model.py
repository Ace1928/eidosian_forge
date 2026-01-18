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
def _sync_final_model(self, is_last_joiner):
    self._authoritative_rank = self._find_common_rank(self._distributed_rank, is_last_joiner)
    _sync_module_states(module=self.module, process_group=self.process_group, broadcast_bucket_size=self.broadcast_bucket_size, src=self._authoritative_rank, params_and_buffers_to_ignore=self.parameters_to_ignore, broadcast_buffers=self.broadcast_buffers)