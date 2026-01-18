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
def _register_autograd_kernel(self):
    assert self._has_impl('backward')
    assert self._has_impl('save_for_backward')
    kernel = construct_autograd_kernel(self._schema, self._output_differentiability, self, get_op(self._qualname), self._get_impl('save_for_backward').func, self._get_impl('backward').func)
    self._register_impl('autograd', kernel)