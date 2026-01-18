import contextlib
import ctypes
import importlib
import inspect
import sys
import types
from typing import Any, Callable, Dict, List, Type, Union
import torch._C
import torch.utils._pytree as pytree
from torch import _utils_internal
from torch._functorch.pyfunctorch import dispatch_functorch
class OpOverloadPacket:

    def __init__(self, qualified_op_name, op_name, op, overload_names):
        self._qualified_op_name = qualified_op_name
        self.__name__ = op_name
        self._op = op
        self._overload_names = overload_names
        self._dir = []

    def __deepcopy__(self, memo=None):
        return self

    def __repr__(self):
        return "<OpOverloadPacket(op='{}.{}')>".format(*self._qualified_op_name.split('::'))

    def __hash__(self):
        return hash(self._op)

    def __str__(self):
        return '{}.{}'.format(*self._qualified_op_name.split('::'))

    @property
    def op(self):
        return self._op

    def __getattr__(self, key):
        if key == '__file__':
            return 'torch.ops'
        try:
            if key.startswith('__'):
                return getattr(self._op, key)
        except AttributeError:
            raise AttributeError(f"'{str(self)}' can't have an overload name beginning with '__' and the underlying op {str(self._op)} has no attribute {key} either.") from None
        try:
            use_key = '' if key == 'default' else key
            op_, op_dk_, tags = torch._C._get_operation_overload(self._qualified_op_name, use_key)
            schema = torch._C._get_schema(self._qualified_op_name, use_key)
            overload = OpOverload(self, op_, op_dk_, schema, tags)
            setattr(self, key, overload)
            self._dir.append(key)
            return overload
        except RuntimeError:
            raise AttributeError(f"The underlying op of '{str(self)}' has no overload name '{key}'") from None

    def __iter__(self):
        return iter(self._dir)

    def __call__(self, *args, **kwargs):
        return self._op(*args, **kwargs or {})

    def overloads(self):
        return [n if n else 'default' for n in self._overload_names]