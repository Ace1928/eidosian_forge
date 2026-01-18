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
def _check_comm_hook(self, hook):
    if not callable(hook):
        self._log_and_throw(TypeError, 'Communication hook must be callable.')
    sig = inspect.signature(hook)
    if sig.parameters['bucket'].annotation != inspect._empty and sig.parameters['bucket'].annotation != dist.GradBucket:
        self._log_and_throw(ValueError, 'Communication hook: bucket annotation should be dist.GradBucket.')
    if sig.return_annotation != inspect._empty and sig.return_annotation != torch.futures.Future[torch.Tensor]:
        self._log_and_throw(ValueError, 'Communication hook: return annotation should be torch.futures.Future[torch.Tensor].')
    if hook.__name__ in ['bf16_compress_hook', 'bf16_compress_wrapper_hook'] and (torch.version.cuda is None and torch.version.hip is None or (torch.version.cuda is not None and int(torch.version.cuda.split('.')[0]) < 11) or (not dist.is_available()) or (not dist.is_nccl_available()) or (torch.cuda.nccl.version() < (2, 10))):
        self._log_and_throw(TypeError, 'BF16 all reduce communication hook required CUDA 11+ and NCCL 2.10+.')