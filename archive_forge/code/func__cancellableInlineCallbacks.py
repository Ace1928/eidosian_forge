from __future__ import annotations
import inspect
import traceback
import warnings
from abc import ABC, abstractmethod
from asyncio import AbstractEventLoop, Future, iscoroutine
from contextvars import Context as _Context, copy_context as _copy_context
from enum import Enum
from functools import wraps
from sys import exc_info, implementation
from types import CoroutineType, GeneratorType, MappingProxyType, TracebackType
from typing import (
import attr
from incremental import Version
from typing_extensions import Concatenate, Literal, ParamSpec, Self
from twisted.internet.interfaces import IDelayedCall, IReactorTime
from twisted.logger import Logger
from twisted.python import lockfile
from twisted.python.compat import _PYPY, cmp, comparable
from twisted.python.deprecate import deprecated, warnAboutFunction
from twisted.python.failure import Failure, _extraneous
def _cancellableInlineCallbacks(gen: Union[Generator[Deferred[Any], object, _T], Coroutine[Deferred[Any], object, _T]]) -> Deferred[_T]:
    """
    Make an C{@}L{inlineCallbacks} cancellable.

    @param gen: a generator object returned by calling a function or method
        decorated with C{@}L{inlineCallbacks}

    @return: L{Deferred} for the C{@}L{inlineCallbacks} that is cancellable.
    """
    deferred: Deferred[_T] = Deferred(lambda d: _addCancelCallbackToDeferred(d, status))
    status = _CancellationStatus(deferred)
    _inlineCallbacks(None, gen, status, _copy_context())
    return deferred