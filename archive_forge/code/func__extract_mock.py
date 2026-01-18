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
def _extract_mock(obj):
    if isinstance(obj, FunctionTypes) and hasattr(obj, 'mock'):
        return obj.mock
    else:
        return obj