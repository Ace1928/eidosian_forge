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
def _is_async_func(func):
    if getattr(func, '__code__', None):
        return iscoroutinefunction(func)
    else:
        return False