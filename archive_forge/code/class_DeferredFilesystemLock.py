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
class DeferredFilesystemLock(lockfile.FilesystemLock):
    """
    A L{FilesystemLock} that allows for a L{Deferred} to be fired when the lock is
    acquired.

    @ivar _scheduler: The object in charge of scheduling retries. In this
        implementation this is parameterized for testing.
    @ivar _interval: The retry interval for an L{IReactorTime} based scheduler.
    @ivar _tryLockCall: An L{IDelayedCall} based on C{_interval} that will manage
        the next retry for acquiring the lock.
    @ivar _timeoutCall: An L{IDelayedCall} based on C{deferUntilLocked}'s timeout
        argument.  This is in charge of timing out our attempt to acquire the
        lock.
    """
    _interval = 1
    _tryLockCall: Optional[IDelayedCall] = None
    _timeoutCall: Optional[IDelayedCall] = None

    def __init__(self, name: str, scheduler: Optional[IReactorTime]=None) -> None:
        """
        @param name: The name of the lock to acquire
        @param scheduler: An object which provides L{IReactorTime}
        """
        lockfile.FilesystemLock.__init__(self, name)
        if scheduler is None:
            from twisted.internet import reactor
            scheduler = cast(IReactorTime, reactor)
        self._scheduler = scheduler

    def deferUntilLocked(self, timeout: Optional[float]=None) -> Deferred[None]:
        """
        Wait until we acquire this lock.  This method is not safe for
        concurrent use.

        @param timeout: the number of seconds after which to time out if the
            lock has not been acquired.

        @return: a L{Deferred} which will callback when the lock is acquired, or
            errback with a L{TimeoutError} after timing out or an
            L{AlreadyTryingToLockError} if the L{deferUntilLocked} has already
            been called and not successfully locked the file.
        """
        if self._tryLockCall is not None:
            return fail(AlreadyTryingToLockError("deferUntilLocked isn't safe for concurrent use."))

        def _cancelLock(reason: Union[Failure, Exception]) -> None:
            """
            Cancel a L{DeferredFilesystemLock.deferUntilLocked} call.

            @type reason: L{Failure}
            @param reason: The reason why the call is cancelled.
            """
            assert self._tryLockCall is not None
            self._tryLockCall.cancel()
            self._tryLockCall = None
            if self._timeoutCall is not None and self._timeoutCall.active():
                self._timeoutCall.cancel()
                self._timeoutCall = None
            if self.lock():
                d.callback(None)
            else:
                d.errback(reason)
        d: Deferred[None] = Deferred(lambda deferred: _cancelLock(CancelledError()))

        def _tryLock() -> None:
            if self.lock():
                if self._timeoutCall is not None:
                    self._timeoutCall.cancel()
                    self._timeoutCall = None
                self._tryLockCall = None
                d.callback(None)
            else:
                if timeout is not None and self._timeoutCall is None:
                    reason = Failure(TimeoutError('Timed out acquiring lock: %s after %fs' % (self.name, timeout)))
                    self._timeoutCall = self._scheduler.callLater(timeout, _cancelLock, reason)
                self._tryLockCall = self._scheduler.callLater(self._interval, _tryLock)
        _tryLock()
        return d