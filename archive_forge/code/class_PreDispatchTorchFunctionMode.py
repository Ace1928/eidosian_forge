import contextlib
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
from torch.fx import Tracer, GraphModule
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, unset_fake_temporarily, is_fake
from torch._dispatch.python import enable_python_dispatcher, enable_pre_dispatch
import torch.fx as fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from contextlib import contextmanager, nullcontext
import inspect
from dataclasses import dataclass
import weakref
import operator
from torch.utils._stats import count
import logging
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import (
from .sym_node import SymNode
from ._sym_dispatch_mode import SymDispatchMode
from torch.fx import Proxy
import torch.fx.traceback as fx_traceback
from torch import SymInt, SymFloat, SymBool
from torch.utils.weak import WeakTensorKeyDictionary
class PreDispatchTorchFunctionMode(TorchFunctionMode):

    def __init__(self, tracer):
        self.tracer = tracer

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        pre_dispatch_ops = [torch._C._set_grad_enabled, torch.amp._enter_autocast, torch.amp._exit_autocast]
        if func in pre_dispatch_ops:
            return self.tracer.create_node('call_function', func, args, {})
        return func(*args, **kwargs)