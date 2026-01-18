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
def _default_broadcast_coalesced(self, bufs=None, bucket_size=None, authoritative_rank=0):
    """
        Broadcasts buffers from rank 0 to rest of workers.

        If bufs, bucket_size are None, default values self.modules_buffers
        and self.broadcast_bucket_size are used instead.
        """
    if bufs is None:
        bufs = self.modules_buffers
    if bucket_size is None:
        bucket_size = self.broadcast_bucket_size
    self._distributed_broadcast_coalesced(bufs, bucket_size, authoritative_rank)