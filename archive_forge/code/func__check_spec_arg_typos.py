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
def _check_spec_arg_typos(kwargs_to_check):
    typos = ('autospect', 'auto_spec', 'set_spec')
    for typo in typos:
        if typo in kwargs_to_check:
            raise RuntimeError(f'{typo!r} might be a typo; use unsafe=True if this is intended')