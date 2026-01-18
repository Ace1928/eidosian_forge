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
def _get_call_arguments(self):
    if len(self) == 2:
        args, kwargs = self
    else:
        name, args, kwargs = self
    return (args, kwargs)