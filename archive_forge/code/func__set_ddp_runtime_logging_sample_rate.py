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
def _set_ddp_runtime_logging_sample_rate(self, sample_rate):
    """
        Set sample_rate of collecting runtime stats.

        This interface allows users to set sample_rate of collecting
        runtime stats. The runtime stats will be recorded for the
        first 10 iterations, after 10 iterations runtime stats will be
        recorded once every "sample_rate" training iterations. In
        default, runtime stats are recorded for the first 10 iterations,
        after 10 iterations runtime stats are recorded once every
        "kDDPRuntimeLoggingSampleRate=100" training iterations.
        This is a prototype interface and subject to change in the future.
        """
    if sample_rate < 1:
        self._log_and_throw(ValueError, 'DDP runtime logging sample rate should be equal or greater than 1')
    self.reducer._set_ddp_runtime_logging_sample_rate(sample_rate)