import asyncio
import contextlib
import io
import inspect
import pprint
import sys
import builtins
import pkgutil
from asyncio import iscoroutinefunction
from types import CodeType, ModuleType, MethodType
from unittest.util import safe_repr
from functools import wraps, partial
from threading import RLock
def _is_async_obj(obj):
    if _is_instance_mock(obj) and (not isinstance(obj, AsyncMock)):
        return False
    if hasattr(obj, '__func__'):
        obj = getattr(obj, '__func__')
    return iscoroutinefunction(obj) or inspect.isawaitable(obj)