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
class ProxyTorchDispatchMode(TorchDispatchMode):

    def __init__(self, tracer, tracing_mode, pre_dispatch=False, _allow_fake_constant=False, _error_on_data_dependent_ops=True):
        dk = torch._C.DispatchKey.PreDispatch if pre_dispatch else None
        super().__init__(dk)
        self.tracer = tracer
        self.tracing_mode = tracing_mode
        self.enable_tracing = True
        self.pre_dispatch = pre_dispatch
        self._allow_fake_constant = _allow_fake_constant
        self._error_on_data_dependent_ops = _error_on_data_dependent_ops
        self.sym_mode = ProxySymDispatchMode(tracer)
        self.trace_state = {}
        self._managers = []
        self._mode_key = torch._C._TorchDispatchModeKey.PROXY
        self.enter_stack: List[Optional[ProxyTorchDispatchMode]] = []

    @count
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        with self.sym_mode.enable(False), set_original_aten_op(func):
            return self.inner_torch_dispatch(func, types, args, kwargs)

    def __enter__(self):
        m = self.sym_mode.enable(True)
        self._managers.append(m)
        m.__enter__()
        maybe_prev_proxy_mode = torch._C._unset_dispatch_mode(self._mode_key)
        self.enter_stack.append(maybe_prev_proxy_mode)
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        m = self._managers.pop()
        b = super().__exit__(exc_type, exc_value, traceback)
        mb_previous_proxy_mode = self.enter_stack.pop()
        if mb_previous_proxy_mode is not None:
            torch._C._set_dispatch_mode(mb_previous_proxy_mode)
        if not b:
            return m.__exit__(exc_type, exc_value, traceback)
        else:
            return m.__exit__(None, None, None)

    def inner_torch_dispatch(self, func, types, args=(), kwargs=None):
        if not self.enable_tracing:
            return func(*args, **kwargs)
        if func in [prim.device.default]:
            return func(*args, **kwargs)
        return proxy_call(self, func, self.pre_dispatch, args, kwargs)