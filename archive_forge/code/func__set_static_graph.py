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
def _set_static_graph(self):
    """
        Set static graph for DDP.

        It is recommended to set static graph in the DDP constructor, which will
        call this private API internally.
        """
    if self.static_graph:
        warnings.warn("You've set static_graph to be True, no need to set it again.")
        return
    self.static_graph = True
    self._static_graph_delay_allreduce_enqueued = False
    self.reducer._set_static_graph()
    assert self.logger is not None
    self.logger._set_static_graph()
    if self.find_unused_parameters:
        warnings.warn('You passed find_unused_parameters=true to DistributedDataParallel, `_set_static_graph` will detect unused parameters automatically, so you do not need to set find_unused_parameters=true, just be sure these unused parameters will not change during training loop while calling `_set_static_graph`.')