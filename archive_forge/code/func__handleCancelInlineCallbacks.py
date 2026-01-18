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
def _handleCancelInlineCallbacks(result: Failure, status: _CancellationStatus[_T], /) -> Deferred[_T]:
    """
    Propagate the cancellation of an C{@}L{inlineCallbacks} to the
    L{Deferred} it is waiting on.

    @param result: An L{_InternalInlineCallbacksCancelledError} from
        C{cancel()}.
    @param status: a L{_CancellationStatus} tracking the current status of C{gen}
    @return: A new L{Deferred} that the C{@}L{inlineCallbacks} generator
        can callback or errback through.
    """
    result.trap(_InternalInlineCallbacksCancelledError)
    status.deferred = Deferred(lambda d: _addCancelCallbackToDeferred(d, status))
    awaited = status.waitingOn
    assert awaited is not None
    awaited.cancel()
    return status.deferred