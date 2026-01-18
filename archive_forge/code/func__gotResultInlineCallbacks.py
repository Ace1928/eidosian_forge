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
def _gotResultInlineCallbacks(r: object, waiting: List[Any], gen: Union[Generator[Deferred[Any], Any, _T], Coroutine[Deferred[Any], Any, _T]], status: _CancellationStatus[_T], context: _Context) -> None:
    """
    Helper for L{_inlineCallbacks} to handle a nested L{Deferred} firing.

    @param r: The result of the L{Deferred}
    @param waiting: Whether the L{_inlineCallbacks} was waiting, and the result.
    @param gen: a generator object returned by calling a function or method
        decorated with C{@}L{inlineCallbacks}
    @param status: a L{_CancellationStatus} tracking the current status of C{gen}
    @param context: the contextvars context to run `gen` in
    """
    if waiting[0]:
        waiting[0] = False
        waiting[1] = r
    else:
        _inlineCallbacks(r, gen, status, context)