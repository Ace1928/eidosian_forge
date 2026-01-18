from __future__ import annotations
import asyncio
import inspect
import math
import operator
from collections.abc import Iterable, Iterator
from functools import partial
from types import FunctionType, MethodType
from typing import Any, Callable, Optional
from .depends import depends
from .display import _display_accessors, _reactive_display_objs
from .parameterized import (
from .parameters import Boolean, Event
from ._utils import _to_async_gen, iscoroutinefunction, full_groupby
def _eval_operation(self, obj, operation):
    fn, args, kwargs = (operation['fn'], operation['args'], operation['kwargs'])
    resolved_args = []
    for arg in args:
        val = resolve_value(arg)
        if val is Skip or val is Undefined:
            raise Skip
        resolved_args.append(val)
    resolved_kwargs = {}
    for k, arg in kwargs.items():
        val = resolve_value(arg)
        if val is Skip or val is Undefined:
            raise Skip
        resolved_kwargs[k] = val
    if isinstance(fn, str):
        obj = getattr(obj, fn)(*resolved_args, **resolved_kwargs)
    elif operation.get('reverse'):
        obj = fn(resolved_args[0], obj, *resolved_args[1:], **resolved_kwargs)
    else:
        obj = fn(obj, *resolved_args, **resolved_kwargs)
    return obj