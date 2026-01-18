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
def _check_doesnt_have_library_autograd_impl(self):
    if self._registered_autograd_kernel_indirection:
        return
    if _C._dispatch_has_kernel_for_dispatch_key(self._qualname, 'CompositeImplicitAutograd'):
        raise RuntimeError(f'impl_backward/impl_save_for_backward: the operator {self._qualname} already has an implementation for this device type via a pre-existing registration to DispatchKey::CompositeImplicitAutograd.CompositeImplicitAutograd operators do not need an autograd formula; instead, the operator will decompose into its constituents and those can have autograd formulas defined on them.')
    for key in ['Autograd', 'AutogradCPU', 'AutogradCUDA']:
        if _C._dispatch_has_kernel_for_dispatch_key(self._qualname, key):
            raise RuntimeError(f"impl_backward/impl_save_for_backward: the operator {self._qualname} already has an Autograd kernel registered to DispatchKey::{key} vi a pre-existing torch.library or TORCH_LIBRARY registration. Please either remove those registrations or don't use the torch._custom_ops APIs")