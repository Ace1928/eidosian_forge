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
def _fire_reducer_autograd_hook(self, idx, *unused):
    """
        Fire the reducer's autograd hook to allreduce params in a Reducer bucket.

        Note that this is only used during mixed precision training as the
        Reducer's hooks installed during construction time would not be called
        as we're working in the low precision parameter setting.
        """
    self.reducer._autograd_hook(idx)