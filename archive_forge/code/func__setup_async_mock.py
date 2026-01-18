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
def _setup_async_mock(mock):
    mock._is_coroutine = asyncio.coroutines._is_coroutine
    mock.await_count = 0
    mock.await_args = None
    mock.await_args_list = _CallList()

    def wrapper(attr, /, *args, **kwargs):
        return getattr(mock.mock, attr)(*args, **kwargs)
    for attribute in ('assert_awaited', 'assert_awaited_once', 'assert_awaited_with', 'assert_awaited_once_with', 'assert_any_await', 'assert_has_awaits', 'assert_not_awaited'):
        setattr(mock, attribute, partial(wrapper, attribute))