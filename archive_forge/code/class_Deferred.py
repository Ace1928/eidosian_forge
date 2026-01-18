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
class Deferred(Awaitable[_SelfResultT]):
    """
    This is a callback which will be put off until later.

    Why do we want this? Well, in cases where a function in a threaded
    program would block until it gets a result, for Twisted it should
    not block. Instead, it should return a L{Deferred}.

    This can be implemented for protocols that run over the network by
    writing an asynchronous protocol for L{twisted.internet}. For methods
    that come from outside packages that are not under our control, we use
    threads (see for example L{twisted.enterprise.adbapi}).

    For more information about Deferreds, see doc/core/howto/defer.html or
    U{http://twistedmatrix.com/documents/current/core/howto/defer.html}

    When creating a Deferred, you may provide a canceller function, which
    will be called by d.cancel() to let you do any clean-up necessary if the
    user decides not to wait for the deferred to complete.

    @ivar called: A flag which is C{False} until either C{callback} or
        C{errback} is called and afterwards always C{True}.
    @ivar paused: A counter of how many unmatched C{pause} calls have been made
        on this instance.
    @ivar _suppressAlreadyCalled: A flag used by the cancellation mechanism
        which is C{True} if the Deferred has no canceller and has been
        cancelled, C{False} otherwise.  If C{True}, it can be expected that
        C{callback} or C{errback} will eventually be called and the result
        should be silently discarded.
    @ivar _runningCallbacks: A flag which is C{True} while this instance is
        executing its callback chain, used to stop recursive execution of
        L{_runCallbacks}
    @ivar _chainedTo: If this L{Deferred} is waiting for the result of another
        L{Deferred}, this is a reference to the other Deferred.  Otherwise,
        L{None}.
    """
    called = False
    paused = 0
    _debugInfo: Optional[DebugInfo] = None
    _suppressAlreadyCalled = False
    _runningCallbacks = False
    debug = False
    _chainedTo: 'Optional[Deferred[Any]]' = None

    def __init__(self, canceller: Optional[Callable[['Deferred[Any]'], None]]=None) -> None:
        """
        Initialize a L{Deferred}.

        @param canceller: a callable used to stop the pending operation
            scheduled by this L{Deferred} when L{Deferred.cancel} is invoked.
            The canceller will be passed the deferred whose cancellation is
            requested (i.e., C{self}).

            If a canceller is not given, or does not invoke its argument's
            C{callback} or C{errback} method, L{Deferred.cancel} will
            invoke L{Deferred.errback} with a L{CancelledError}.

            Note that if a canceller is not given, C{callback} or
            C{errback} may still be invoked exactly once, even though
            defer.py will have already invoked C{errback}, as described
            above.  This allows clients of code which returns a L{Deferred}
            to cancel it without requiring the L{Deferred} instantiator to
            provide any specific implementation support for cancellation.
            New in 10.1.

        @type canceller: a 1-argument callable which takes a L{Deferred}. The
            return result is ignored.
        """
        self.callbacks: List[_CallbackChain] = []
        self._canceller = canceller
        if self.debug:
            self._debugInfo = DebugInfo()
            self._debugInfo.creator = traceback.format_stack()[:-1]

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

    @overload
    def addCallback(self, callback: Callable[Concatenate[_SelfResultT, _P], Failure], *args: _P.args, **kwargs: _P.kwargs) -> Deferred[_NextResultT]:
        ...

    @overload
    def addCallback(self, callback: Callable[Concatenate[_SelfResultT, _P], Union[Failure, Deferred[_NextResultT]]], *args: _P.args, **kwargs: _P.kwargs) -> Deferred[_NextResultT]:
        ...

    @overload
    def addCallback(self, callback: Callable[Concatenate[_SelfResultT, _P], Union[Failure, _NextResultT]], *args: _P.args, **kwargs: _P.kwargs) -> Deferred[_NextResultT]:
        ...

    @overload
    def addCallback(self, callback: Callable[Concatenate[_SelfResultT, _P], Deferred[_NextResultT]], *args: _P.args, **kwargs: _P.kwargs) -> Deferred[_NextResultT]:
        ...

    @overload
    def addCallback(self, callback: Callable[Concatenate[_SelfResultT, _P], Union[Deferred[_NextResultT], _NextResultT]], *args: _P.args, **kwargs: _P.kwargs) -> Deferred[_NextResultT]:
        ...

    @overload
    def addCallback(self, callback: Callable[Concatenate[_SelfResultT, _P], _NextResultT], *args: _P.args, **kwargs: _P.kwargs) -> Deferred[_NextResultT]:
        ...

    def addCallback(self, callback: Any, *args: Any, **kwargs: Any) -> 'Deferred[Any]':
        """
        Convenience method for adding just a callback.

        See L{addCallbacks}.
        """
        return self.addCallbacks(callback, callbackArgs=args, callbackKeywords=kwargs)

    @overload
    def addErrback(self, errback: Callable[Concatenate[Failure, _P], Deferred[_NextResultT]], *args: _P.args, **kwargs: _P.kwargs) -> 'Deferred[Union[_SelfResultT, _NextResultT]]':
        ...

    @overload
    def addErrback(self, errback: Callable[Concatenate[Failure, _P], Failure], *args: _P.args, **kwargs: _P.kwargs) -> 'Deferred[Union[_SelfResultT]]':
        ...

    @overload
    def addErrback(self, errback: Callable[Concatenate[Failure, _P], _NextResultT], *args: _P.args, **kwargs: _P.kwargs) -> 'Deferred[Union[_SelfResultT, _NextResultT]]':
        ...

    def addErrback(self, errback: Any, *args: Any, **kwargs: Any) -> 'Deferred[Any]':
        """
        Convenience method for adding just an errback.

        See L{addCallbacks}.
        """
        return self.addCallbacks(passthru, errback, errbackArgs=args, errbackKeywords=kwargs)

    @overload
    def addBoth(self, callback: Callable[Concatenate[Union[_SelfResultT, Failure], _P], Failure], *args: _P.args, **kwargs: _P.kwargs) -> Deferred[_NextResultT]:
        ...

    @overload
    def addBoth(self, callback: Callable[Concatenate[Union[_SelfResultT, Failure], _P], Union[Failure, Deferred[_NextResultT]]], *args: _P.args, **kwargs: _P.kwargs) -> Deferred[_NextResultT]:
        ...

    @overload
    def addBoth(self, callback: Callable[Concatenate[Union[_SelfResultT, Failure], _P], Union[Failure, _NextResultT]], *args: _P.args, **kwargs: _P.kwargs) -> Deferred[_NextResultT]:
        ...

    @overload
    def addBoth(self, callback: Callable[Concatenate[Union[_SelfResultT, Failure], _P], Deferred[_NextResultT]], *args: _P.args, **kwargs: _P.kwargs) -> Deferred[_NextResultT]:
        ...

    @overload
    def addBoth(self, callback: Callable[Concatenate[Union[_SelfResultT, Failure], _P], Union[Deferred[_NextResultT], _NextResultT]], *args: _P.args, **kwargs: _P.kwargs) -> Deferred[_NextResultT]:
        ...

    @overload
    def addBoth(self, callback: Callable[Concatenate[Union[_SelfResultT, Failure], _P], _NextResultT], *args: _P.args, **kwargs: _P.kwargs) -> Deferred[_NextResultT]:
        ...

    @overload
    def addBoth(self, callback: Callable[Concatenate[_T, _P], _T], *args: _P.args, **kwargs: _P.kwargs) -> Deferred[_SelfResultT]:
        ...

    def addBoth(self, callback: Any, *args: Any, **kwargs: Any) -> 'Deferred[Any]':
        """
        Convenience method for adding a single callable as both a callback
        and an errback.

        See L{addCallbacks}.
        """
        return self.addCallbacks(callback, callback, callbackArgs=args, errbackArgs=args, callbackKeywords=kwargs, errbackKeywords=kwargs)

    def addTimeout(self, timeout: float, clock: IReactorTime, onTimeoutCancel: Optional[Callable[[Union[_SelfResultT, Failure], float], Union[_NextResultT, Failure]]]=None) -> 'Deferred[Union[_SelfResultT, _NextResultT]]':
        """
        Time out this L{Deferred} by scheduling it to be cancelled after
        C{timeout} seconds.

        The timeout encompasses all the callbacks and errbacks added to this
        L{defer.Deferred} before the call to L{addTimeout}, and none added
        after the call.

        If this L{Deferred} gets timed out, it errbacks with a L{TimeoutError},
        unless a cancelable function was passed to its initialization or unless
        a different C{onTimeoutCancel} callable is provided.

        @param timeout: number of seconds to wait before timing out this
            L{Deferred}
        @param clock: The object which will be used to schedule the timeout.
        @param onTimeoutCancel: A callable which is called immediately after
            this L{Deferred} times out, and not if this L{Deferred} is
            otherwise cancelled before the timeout. It takes an arbitrary
            value, which is the value of this L{Deferred} at that exact point
            in time (probably a L{CancelledError} L{Failure}), and the
            C{timeout}.  The default callable (if C{None} is provided) will
            translate a L{CancelledError} L{Failure} into a L{TimeoutError}.

        @return: C{self}.

        @since: 16.5
        """
        timedOut = [False]

        def timeItOut() -> None:
            timedOut[0] = True
            self.cancel()
        delayedCall = clock.callLater(timeout, timeItOut)

        def convertCancelled(result: Union[_SelfResultT, Failure]) -> Union[_SelfResultT, _NextResultT, Failure]:
            if timedOut[0]:
                toCall = onTimeoutCancel or _cancelledToTimedOutError
                return toCall(result, timeout)
            return result

        def cancelTimeout(result: _T) -> _T:
            if delayedCall.active():
                delayedCall.cancel()
            return result
        converted: Deferred[Union[_SelfResultT, _NextResultT]] = self.addBoth(convertCancelled)
        return converted.addBoth(cancelTimeout)

    def chainDeferred(self, d: 'Deferred[_SelfResultT]') -> 'Deferred[None]':
        """
        Chain another L{Deferred} to this L{Deferred}.

        This method adds callbacks to this L{Deferred} to call C{d}'s callback
        or errback, as appropriate. It is merely a shorthand way of performing
        the following::

            d1.addCallbacks(d2.callback, d2.errback)

        When you chain a deferred C{d2} to another deferred C{d1} with
        C{d1.chainDeferred(d2)}, you are making C{d2} participate in the
        callback chain of C{d1}.
        Thus any event that fires C{d1} will also fire C{d2}.
        However, the converse is B{not} true; if C{d2} is fired, C{d1} will not
        be affected.

        Note that unlike the case where chaining is caused by a L{Deferred}
        being returned from a callback, it is possible to cause the call
        stack size limit to be exceeded by chaining many L{Deferred}s
        together with C{chainDeferred}.

        @return: C{self}.
        """
        d._chainedTo = self
        return self.addCallbacks(d.callback, d.errback)

    def callback(self, result: Union[_SelfResultT, Failure]) -> None:
        """
        Run all success callbacks that have been added to this L{Deferred}.

        Each callback will have its result passed as the first argument to
        the next; this way, the callbacks act as a 'processing chain'.  If
        the success-callback returns a L{Failure} or raises an L{Exception},
        processing will continue on the *error* callback chain.  If a
        callback (or errback) returns another L{Deferred}, this L{Deferred}
        will be chained to it (and further callbacks will not run until that
        L{Deferred} has a result).

        An instance of L{Deferred} may only have either L{callback} or
        L{errback} called on it, and only once.

        @param result: The object which will be passed to the first callback
            added to this L{Deferred} (via L{addCallback}), unless C{result} is
            a L{Failure}, in which case the behavior is the same as calling
            C{errback(result)}.

        @raise AlreadyCalledError: If L{callback} or L{errback} has already been
            called on this L{Deferred}.
        """
        assert not isinstance(result, Deferred)
        self._startRunCallbacks(result)

    def errback(self, fail: Optional[Union[Failure, BaseException]]=None) -> None:
        """
        Run all error callbacks that have been added to this L{Deferred}.

        Each callback will have its result passed as the first
        argument to the next; this way, the callbacks act as a
        'processing chain'. Also, if the error-callback returns a non-Failure
        or doesn't raise an L{Exception}, processing will continue on the
        *success*-callback chain.

        If the argument that's passed to me is not a L{Failure} instance,
        it will be embedded in one. If no argument is passed, a
        L{Failure} instance will be created based on the current
        traceback stack.

        Passing a string as `fail' is deprecated, and will be punished with
        a warning message.

        An instance of L{Deferred} may only have either L{callback} or
        L{errback} called on it, and only once.

        @param fail: The L{Failure} object which will be passed to the first
            errback added to this L{Deferred} (via L{addErrback}).
            Alternatively, a L{Exception} instance from which a L{Failure} will
            be constructed (with no traceback) or L{None} to create a L{Failure}
            instance from the current exception state (with a traceback).

        @raise AlreadyCalledError: If L{callback} or L{errback} has already been
            called on this L{Deferred}.
        @raise NoCurrentExceptionError: If C{fail} is L{None} but there is
            no current exception state.
        """
        if fail is None:
            fail = Failure(captureVars=self.debug)
        elif not isinstance(fail, Failure):
            fail = Failure(fail)
        self._startRunCallbacks(fail)

    def pause(self) -> None:
        """
        Stop processing on a L{Deferred} until L{unpause}() is called.
        """
        self.paused = self.paused + 1

    def unpause(self) -> None:
        """
        Process all callbacks made since L{pause}() was called.
        """
        self.paused = self.paused - 1
        if self.paused:
            return
        if self.called:
            self._runCallbacks()

    def cancel(self) -> None:
        """
        Cancel this L{Deferred}.

        If the L{Deferred} has not yet had its C{errback} or C{callback} method
        invoked, call the canceller function provided to the constructor. If
        that function does not invoke C{callback} or C{errback}, or if no
        canceller function was provided, errback with L{CancelledError}.

        If this L{Deferred} is waiting on another L{Deferred}, forward the
        cancellation to the other L{Deferred}.
        """
        if not self.called:
            canceller = self._canceller
            if canceller:
                canceller(self)
            else:
                self._suppressAlreadyCalled = True
            if not self.called:
                self.errback(Failure(CancelledError()))
        elif isinstance(self.result, Deferred):
            self.result.cancel()

    def _startRunCallbacks(self, result: object) -> None:
        if self.called:
            if self._suppressAlreadyCalled:
                self._suppressAlreadyCalled = False
                return
            if self.debug:
                if self._debugInfo is None:
                    self._debugInfo = DebugInfo()
                extra = '\n' + self._debugInfo._getDebugTracebacks()
                raise AlreadyCalledError(extra)
            raise AlreadyCalledError
        if self.debug:
            if self._debugInfo is None:
                self._debugInfo = DebugInfo()
            self._debugInfo.invoker = traceback.format_stack()[:-2]
        self.called = True
        self._canceller = None
        self.result = result
        self._runCallbacks()

    def _continuation(self) -> _CallbackChain:
        """
        Build a tuple of callback and errback with L{_Sentinel._CONTINUE}.
        """
        return ((_Sentinel._CONTINUE, (self,), _NONE_KWARGS), (_Sentinel._CONTINUE, (self,), _NONE_KWARGS))

    def _runCallbacks(self) -> None:
        """
        Run the chain of callbacks once a result is available.

        This consists of a simple loop over all of the callbacks, calling each
        with the current result and making the current result equal to the
        return value (or raised exception) of that call.

        If L{_runningCallbacks} is true, this loop won't run at all, since
        it is already running above us on the call stack.  If C{self.paused} is
        true, the loop also won't run, because that's what it means to be
        paused.

        The loop will terminate before processing all of the callbacks if a
        L{Deferred} without a result is encountered.

        If a L{Deferred} I{with} a result is encountered, that result is taken
        and the loop proceeds.

        @note: The implementation is complicated slightly by the fact that
            chaining (associating two L{Deferred}s with each other such that one
            will wait for the result of the other, as happens when a Deferred is
            returned from a callback on another L{Deferred}) is supported
            iteratively rather than recursively, to avoid running out of stack
            frames when processing long chains.
        """
        if self._runningCallbacks:
            return
        chain: List[Deferred[Any]] = [self]
        while chain:
            current = chain[-1]
            if current.paused:
                return
            finished = True
            current._chainedTo = None
            while current.callbacks:
                item = current.callbacks.pop(0)
                if not isinstance(current.result, Failure):
                    callback, args, kwargs = item[0]
                else:
                    callback, args, kwargs = item[1]
                if callback is _CONTINUE:
                    chainee = cast(Deferred[object], args[0])
                    chainee.result = current.result
                    current.result = None
                    if current._debugInfo is not None:
                        current._debugInfo.failResult = None
                    chainee.paused -= 1
                    chain.append(chainee)
                    finished = False
                    break
                try:
                    current._runningCallbacks = True
                    try:
                        current.result = callback(current.result, *args, **kwargs)
                        if current.result is current:
                            warnAboutFunction(callback, 'Callback returned the Deferred it was attached to; this breaks the callback chain and will raise an exception in the future.')
                    finally:
                        current._runningCallbacks = False
                except BaseException:
                    current.result = Failure(captureVars=self.debug)
                else:
                    if isinstance(current.result, Deferred):
                        resultResult = getattr(current.result, 'result', _NO_RESULT)
                        if resultResult is _NO_RESULT or isinstance(resultResult, Deferred) or current.result.paused:
                            current.pause()
                            current._chainedTo = current.result
                            current.result.callbacks.append(current._continuation())
                            break
                        else:
                            current.result.result = None
                            if current.result._debugInfo is not None:
                                current.result._debugInfo.failResult = None
                            current.result = resultResult
            if finished:
                if isinstance(current.result, Failure):
                    current.result.cleanFailure()
                    if current._debugInfo is None:
                        current._debugInfo = DebugInfo()
                    current._debugInfo.failResult = current.result
                elif current._debugInfo is not None:
                    current._debugInfo.failResult = None
                chain.pop()

    def __str__(self) -> str:
        """
        Return a string representation of this L{Deferred}.
        """
        cname = self.__class__.__name__
        result = getattr(self, 'result', _NO_RESULT)
        myID = id(self)
        if self._chainedTo is not None:
            result = f' waiting on Deferred at 0x{id(self._chainedTo):x}'
        elif result is _NO_RESULT:
            result = ''
        else:
            result = f' current result: {result!r}'
        return f'<{cname} at 0x{myID:x}{result}>'
    __repr__ = __str__

    def __iter__(self) -> 'Deferred[_SelfResultT]':
        return self

    @_extraneous
    def send(self, value: object=None) -> 'Deferred[_SelfResultT]':
        if self.paused:
            return self
        result = getattr(self, 'result', _NO_RESULT)
        if result is _NO_RESULT:
            return self
        if isinstance(result, Failure):
            assert self._debugInfo is not None
            self._debugInfo.failResult = None
            result.value.__failure__ = result
            raise result.value
        else:
            raise StopIteration(result)

    def __await__(self) -> Generator[Any, None, _SelfResultT]:
        return self.__iter__()
    __next__ = send

    def asFuture(self, loop: AbstractEventLoop) -> 'Future[_SelfResultT]':
        """
        Adapt this L{Deferred} into a L{Future} which is bound to C{loop}.

        @note: converting a L{Deferred} to an L{Future} consumes both
            its result and its errors, so this method implicitly converts
            C{self} into a L{Deferred} firing with L{None}, regardless of what
            its result previously would have been.

        @since: Twisted 17.5.0

        @param loop: The L{asyncio} event loop to bind the L{Future} to.

        @return: A L{Future} which will fire when the L{Deferred} fires.
        """
        future = loop.create_future()

        def checkCancel(futureAgain: 'Future[_SelfResultT]') -> None:
            if futureAgain.cancelled():
                self.cancel()

        def maybeFail(failure: Failure) -> None:
            if not future.cancelled():
                future.set_exception(failure.value)

        def maybeSucceed(result: object) -> None:
            if not future.cancelled():
                future.set_result(result)
        self.addCallbacks(maybeSucceed, maybeFail)
        future.add_done_callback(checkCancel)
        return future

    @classmethod
    def fromFuture(cls, future: 'Future[_SelfResultT]') -> 'Deferred[_SelfResultT]':
        """
        Adapt a L{Future} to a L{Deferred}.

        @note: This creates a L{Deferred} from a L{Future}, I{not} from
            a C{coroutine}; in other words, you will need to call
            L{asyncio.ensure_future}, L{asyncio.loop.create_task} or create an
            L{asyncio.Task} yourself to get from a C{coroutine} to a
            L{Future} if what you have is an awaitable coroutine and
            not a L{Future}.  (The length of this list of techniques is
            exactly why we have left it to the caller!)

        @since: Twisted 17.5.0

        @param future: The L{Future} to adapt.

        @return: A L{Deferred} which will fire when the L{Future} fires.
        """

        def adapt(result: Future[_SelfResultT]) -> None:
            try:
                extracted: _SelfResultT | Failure = result.result()
            except BaseException:
                extracted = Failure()
            actual.callback(extracted)
        futureCancel = object()

        def cancel(reself: Deferred[object]) -> None:
            future.cancel()
            reself.callback(futureCancel)
        self = cls(cancel)
        actual = self

        def uncancel(result: _SelfResultT) -> Union[_SelfResultT, Deferred[_SelfResultT]]:
            if result is futureCancel:
                nonlocal actual
                actual = Deferred()
                return actual
            return result
        self.addCallback(uncancel)
        future.add_done_callback(adapt)
        return self

    @classmethod
    def fromCoroutine(cls, coro: Union[Coroutine[Deferred[Any], Any, _T], Generator[Deferred[Any], Any, _T]]) -> 'Deferred[_T]':
        """
        Schedule the execution of a coroutine that awaits on L{Deferred}s,
        wrapping it in a L{Deferred} that will fire on success/failure of the
        coroutine.

        Coroutine functions return a coroutine object, similar to how
        generators work. This function turns that coroutine into a Deferred,
        meaning that it can be used in regular Twisted code. For example::

            import treq
            from twisted.internet.defer import Deferred
            from twisted.internet.task import react

            async def crawl(pages):
                results = {}
                for page in pages:
                    results[page] = await treq.content(await treq.get(page))
                return results

            def main(reactor):
                pages = [
                    "http://localhost:8080"
                ]
                d = Deferred.fromCoroutine(crawl(pages))
                d.addCallback(print)
                return d

            react(main)

        @since: Twisted 21.2.0

        @param coro: The coroutine object to schedule.

        @raise ValueError: If C{coro} is not a coroutine or generator.
        """
        if iscoroutine(coro) or inspect.isgenerator(coro):
            return _cancellableInlineCallbacks(coro)
        raise NotACoroutineError(f'{coro!r} is not a coroutine')