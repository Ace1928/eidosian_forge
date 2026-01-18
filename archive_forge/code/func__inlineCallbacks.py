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
@_extraneous
def _inlineCallbacks(result: object, gen: Union[Generator[Deferred[Any], Any, _T], Coroutine[Deferred[Any], Any, _T]], status: _CancellationStatus[_T], context: _Context) -> None:
    """
    Carry out the work of L{inlineCallbacks}.

    Iterate the generator produced by an C{@}L{inlineCallbacks}-decorated
    function, C{gen}, C{send()}ing it the results of each value C{yield}ed by
    that generator, until a L{Deferred} is yielded, at which point a callback
    is added to that L{Deferred} to call this function again.

    @param result: The last result seen by this generator.  Note that this is
        never a L{Deferred} - by the time this function is invoked, the
        L{Deferred} has been called back and this will be a particular result
        at a point in its callback chain.

    @param gen: a generator object returned by calling a function or method
        decorated with C{@}L{inlineCallbacks}

    @param status: a L{_CancellationStatus} tracking the current status of C{gen}

    @param context: the contextvars context to run `gen` in
    """
    waiting: List[Any] = [True, None]
    stopIteration: bool = False
    callbackValue: Any = None
    while 1:
        try:
            isFailure = isinstance(result, Failure)
            if isFailure:
                result = context.run(cast(Failure, result).throwExceptionIntoGenerator, gen)
            else:
                result = context.run(gen.send, result)
        except StopIteration as e:
            stopIteration = True
            callbackValue = getattr(e, 'value', None)
        except _DefGen_Return as e:
            excInfo = exc_info()
            assert excInfo is not None
            traceback = excInfo[2]
            assert traceback is not None
            appCodeTrace = traceback.tb_next
            assert appCodeTrace is not None
            if _oldPypyStack:
                appCodeTrace = appCodeTrace.tb_next
                assert appCodeTrace is not None
            if isFailure:
                appCodeTrace = appCodeTrace.tb_next
                assert appCodeTrace is not None
            assert appCodeTrace.tb_next is not None
            if appCodeTrace.tb_next.tb_next:
                ultimateTrace = appCodeTrace
                assert ultimateTrace is not None
                assert ultimateTrace.tb_next is not None
                while ultimateTrace.tb_next.tb_next:
                    ultimateTrace = ultimateTrace.tb_next
                    assert ultimateTrace is not None
                filename = ultimateTrace.tb_frame.f_code.co_filename
                lineno = ultimateTrace.tb_lineno
                assert ultimateTrace.tb_frame is not None
                assert appCodeTrace.tb_frame is not None
                warnings.warn_explicit('returnValue() in %r causing %r to exit: returnValue should only be invoked by functions decorated with inlineCallbacks' % (ultimateTrace.tb_frame.f_code.co_name, appCodeTrace.tb_frame.f_code.co_name), DeprecationWarning, filename, lineno)
            stopIteration = True
            callbackValue = e.value
        except BaseException:
            status.deferred.errback()
            return
        if stopIteration:
            status.deferred.callback(callbackValue)
            return
        if isinstance(result, Deferred):
            result.addBoth(_gotResultInlineCallbacks, waiting, gen, status, context)
            if waiting[0]:
                waiting[0] = False
                status.waitingOn = result
                return
            result = waiting[1]
            waiting[0] = True
            waiting[1] = None