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
def _build_params_for_reducer(self):
    modules_and_parameters = [(module, parameter) for module_name, module in self.module.named_modules() for parameter in [param for param_name, param in module.named_parameters(recurse=False) if param.requires_grad and f'{module_name}.{param_name}' not in self.parameters_to_ignore]]
    memo = set()
    modules_and_parameters = [(m, p) for m, p in modules_and_parameters if p not in memo and (not memo.add(p))]
    parameters = [parameter for _, parameter in modules_and_parameters]

    def produces_sparse_gradient(module):
        if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
            return module.sparse
        return False
    expect_sparse_gradient = [produces_sparse_gradient(module) for module, _ in modules_and_parameters]
    self._assign_modules_buffers()
    return (parameters, expect_sparse_gradient)