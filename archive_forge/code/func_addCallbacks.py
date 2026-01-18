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
def addCallbacks(self, callback: Union[Callable[..., _NextResultT], Callable[..., Deferred[_NextResultT]], Callable[..., Failure], Callable[..., Union[_NextResultT, Deferred[_NextResultT], Failure]]], errback: Union[Callable[..., _NextResultT], Callable[..., Deferred[_NextResultT]], Callable[..., Failure], Callable[..., Union[_NextResultT, Deferred[_NextResultT], Failure]], None]=None, callbackArgs: Tuple[Any, ...]=(), callbackKeywords: Mapping[str, Any]=_NONE_KWARGS, errbackArgs: _CallbackOrderedArguments=(), errbackKeywords: _CallbackKeywordArguments=_NONE_KWARGS) -> 'Deferred[_NextResultT]':
    """
        Add a pair of callbacks (success and error) to this L{Deferred}.

        These will be executed when the 'master' callback is run.

        @note: The signature of this function was designed many years before
            PEP 612; ParamSpec provides no mechanism to annotate parameters
            like C{callbackArgs}; this is therefore inherently less type-safe
            than calling C{addCallback} and C{addErrback} separately.

        @return: C{self}.
        """
    if errback is None:
        errback = _failthru
    if callbackArgs is None:
        callbackArgs = ()
    if callbackKeywords is None:
        callbackKeywords = {}
    if errbackArgs is None:
        errbackArgs = ()
    if errbackKeywords is None:
        errbackKeywords = {}
    assert callable(callback)
    assert callable(errback)
    self.callbacks.append(((callback, callbackArgs, callbackKeywords), (errback, errbackArgs, errbackKeywords)))
    if self.called:
        self._runCallbacks()
    return self