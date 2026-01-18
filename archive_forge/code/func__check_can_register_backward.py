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
def _check_can_register_backward(self):

    def error(detail):
        raise RuntimeError(f'Cannot use torch._custom_ops APIs to register backward formula for {detail}. Got operator {self._qualname} with schema: {schema}')
    schema = self._schema
    if schema.kind() != SchemaKind.functional:
        error('non-functional operator')
    rets = schema.returns
    if not schema.returns:
        error('operator with no returns')
    assert len(rets) > 0
    is_non_mutating_view = any((r.annotation is not None and (not r.annotation.is_write) for r in rets))
    if is_non_mutating_view:
        error('operator that returns views')
    allowed_return_types = {BaseType(BaseTy.int): 'int', BaseType(BaseTy.SymInt): 'SymInt', BaseType(BaseTy.bool): 'bool', BaseType(BaseTy.float): 'float', BaseType(BaseTy.Tensor): 'Tensor', ListType(BaseType(BaseTy.Tensor), None): 'List[Tensor]'}
    for ret in schema.returns:
        if ret.type in allowed_return_types:
            continue
        error(f'operator with return not in {list(allowed_return_types.values())} (got {ret.type})')