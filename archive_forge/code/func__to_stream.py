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
def _to_stream(read_data):
    if isinstance(read_data, bytes):
        return io.BytesIO(read_data)
    else:
        return io.StringIO(read_data)