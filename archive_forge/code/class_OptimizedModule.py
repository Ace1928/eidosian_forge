from __future__ import annotations
import contextlib
import dis
import functools
import inspect
import logging
import os
import sys
import textwrap
import threading
import traceback
import types
import warnings
from dataclasses import dataclass
from enum import Enum
from os.path import dirname, join
from typing import (
from unittest.mock import patch
import torch
import torch.fx
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch import _guards
from torch._subclasses import fake_tensor
from torch.export import Constraint
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.experimental.symbolic_shapes import (
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.nn.parallel.distributed import DistributedDataParallel
from ..fx import GraphModule
from .backends.registry import CompilerFn, lookup_backend
from .hooks import Hooks
from . import config, convert_frame, external_utils, skipfiles, utils
from .code_context import code_context
from .exc import CondOpArgsMismatchError, UserError, UserErrorType
from .mutation_guard import install_generation_tagging_init
from .types import CacheEntry, DynamoCallback
from .utils import compile_times
from torch._dispatch.python import enable_python_dispatcher
from torch.utils._python_dispatch import _disable_current_modes
import sympy
class OptimizedModule(torch.nn.Module):
    """
    Wraps the original nn.Module object and later patches its
    forward method to optimized self.forward method.
    """
    _torchdynamo_orig_callable: Callable[..., Any]
    get_compiler_config: Callable[[], Any]

    def __init__(self, mod: torch.nn.Module, dynamo_ctx):
        super().__init__()
        self._orig_mod = mod
        self.dynamo_ctx = dynamo_ctx
        self._initialize()

    def _initialize(self):
        if isinstance(self._orig_mod.forward, types.MethodType) and skipfiles.check(self._orig_mod.forward):
            self.forward = self.dynamo_ctx(external_utils.wrap_inline(self._orig_mod))
        else:
            self.forward = self.dynamo_ctx(self._orig_mod.__call__)
        if hasattr(self._orig_mod, '_initialize_hook'):
            self._forward = self.forward
            self.forward = self._call_lazy_check

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop('forward', None)
        state.pop('__call__', None)
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._initialize()

    def __getattr__(self, name):
        if name == '_orig_mod':
            return self._modules['_orig_mod']
        return getattr(self._orig_mod, name)

    def _call_lazy_check(self, *args, **kwargs):
        if hasattr(self._orig_mod, '_initialize_hook'):
            self._orig_mod._infer_parameters(self._orig_mod, args, kwargs)
        return self._forward(*args, **kwargs)

    def __dir__(self):
        orig_mod_attrs = self._orig_mod.__dir__()
        return orig_mod_attrs + [attr for attr in super().__dir__() if attr not in orig_mod_attrs]