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
def _check_doesnt_have_library_impl(self, device_type):
    if self._has_impl(device_type):
        return
    key = SUPPORTED_DEVICE_TYPE_TO_KEY[device_type]
    if _C._dispatch_has_computed_kernel_for_dispatch_key(self._qualname, key):
        raise RuntimeError(f'impl(..., device_types={device_type}): the operator {self._qualname} already has an implementation for this device type via a pre-existing torch.library or TORCH_LIBRARY registration.')