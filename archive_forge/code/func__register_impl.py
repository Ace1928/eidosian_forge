import dataclasses
import functools
import inspect
import sys
import typing
import weakref
from torchgen.model import FunctionSchema, OperatorName, SchemaKind, BaseType, ListType, BaseTy
import torch
import torch._C as _C
import torch.library as library
from torch._library.abstract_impl import AbstractImplCtx
from torch.library import get_ctx
from .autograd import autograd_kernel_indirection, construct_autograd_kernel
def _register_impl(self, kind, func, stacklevel=2):
    if self._has_impl(kind):
        func_and_location = self._impls[kind]
        assert func_and_location is not None
        location = func_and_location.location
        raise RuntimeError(f'Attempting to register a {kind} impl for operator {self._qualname} that already has a {kind} impl registered from Python at {location}. This is not supported.')
    frame = inspect.getframeinfo(sys._getframe(stacklevel))
    location = f'{frame.filename}:{frame.lineno}'
    self._impls[kind] = FuncAndLocation(func, location)