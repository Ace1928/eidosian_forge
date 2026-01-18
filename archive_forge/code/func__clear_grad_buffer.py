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
def _clear_grad_buffer(self):
    if self._delay_grad_buffer is not None:
        all_param_grad_none = all((param.grad is None for param in self._delay_all_reduce_params))
        for index, param in enumerate(self._delay_all_reduce_params):
            if param.grad is None:
                param.grad = self._delay_grad_views[index]
                if not all_param_grad_none:
                    param.grad.zero_()
        if all_param_grad_none:
            self._delay_grad_buffer.zero_()