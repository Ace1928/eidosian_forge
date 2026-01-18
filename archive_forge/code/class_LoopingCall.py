import sys
import time
import warnings
from typing import (
from zope.interface import implementer
from incremental import Version
from twisted.internet.base import DelayedCall
from twisted.internet.defer import Deferred, ensureDeferred, maybeDeferred
from twisted.internet.error import ReactorNotRunning
from twisted.internet.interfaces import IDelayedCall, IReactorCore, IReactorTime
from twisted.python import log, reflect
from twisted.python.deprecate import _getDeprecationWarningString
from twisted.python.failure import Failure
class LoopingCall:
    """Call a function repeatedly.

    If C{f} returns a deferred, rescheduling will not take place until the
    deferred has fired. The result value is ignored.

    @ivar f: The function to call.
    @ivar a: A tuple of arguments to pass the function.
    @ivar kw: A dictionary of keyword arguments to pass to the function.
    @ivar clock: A provider of
        L{twisted.internet.interfaces.IReactorTime}.  The default is
        L{twisted.internet.reactor}. Feel free to set this to
        something else, but it probably ought to be set *before*
        calling L{start}.

    @ivar running: A flag which is C{True} while C{f} is scheduled to be called
        (or is currently being called). It is set to C{True} when L{start} is
        called and set to C{False} when L{stop} is called or if C{f} raises an
        exception. In either case, it will be C{False} by the time the
        C{Deferred} returned by L{start} fires its callback or errback.

    @ivar _realLastTime: When counting skips, the time at which the skip
        counter was last invoked.

    @ivar _runAtStart: A flag indicating whether the 'now' argument was passed
        to L{LoopingCall.start}.
    """
    call: Optional[IDelayedCall] = None
    running = False
    _deferred: Optional[Deferred['LoopingCall']] = None
    interval: Optional[float] = None
    _runAtStart = False
    starttime: Optional[float] = None
    _realLastTime: Optional[float] = None

    def __init__(self, f: Callable[..., object], *a: object, **kw: object) -> None:
        self.f = f
        self.a = a
        self.kw = kw
        from twisted.internet import reactor
        self.clock = cast(IReactorTime, reactor)

    @property
    def deferred(self) -> Optional[Deferred['LoopingCall']]:
        """
        DEPRECATED. L{Deferred} fired when loop stops or fails.

        Use the L{Deferred} returned by L{LoopingCall.start}.
        """
        warningString = _getDeprecationWarningString('twisted.internet.task.LoopingCall.deferred', Version('Twisted', 16, 0, 0), replacement='the deferred returned by start()')
        warnings.warn(warningString, DeprecationWarning, stacklevel=2)
        return self._deferred

    @classmethod
    def withCount(cls, countCallable: Callable[[int], object]) -> 'LoopingCall':
        """
        An alternate constructor for L{LoopingCall} that makes available the
        number of calls which should have occurred since it was last invoked.

        Note that this number is an C{int} value; It represents the discrete
        number of calls that should have been made.  For example, if you are
        using a looping call to display an animation with discrete frames, this
        number would be the number of frames to advance.

        The count is normally 1, but can be higher. For example, if the reactor
        is blocked and takes too long to invoke the L{LoopingCall}, a Deferred
        returned from a previous call is not fired before an interval has
        elapsed, or if the callable itself blocks for longer than an interval,
        preventing I{itself} from being called.

        When running with an interval of 0, count will be always 1.

        @param countCallable: A callable that will be invoked each time the
            resulting LoopingCall is run, with an integer specifying the number
            of calls that should have been invoked.

        @return: An instance of L{LoopingCall} with call counting enabled,
            which provides the count as the first positional argument.

        @since: 9.0
        """

        def counter() -> object:
            now = self.clock.seconds()
            if self.interval == 0:
                self._realLastTime = now
                return countCallable(1)
            lastTime = self._realLastTime
            if lastTime is None:
                assert self.starttime is not None, 'LoopingCall called before it was started'
                lastTime = self.starttime
                if self._runAtStart:
                    assert self.interval is not None, 'Looping call called with None interval'
                    lastTime -= self.interval
            lastInterval = self._intervalOf(lastTime)
            thisInterval = self._intervalOf(now)
            count = thisInterval - lastInterval
            if count > 0:
                self._realLastTime = now
                return countCallable(count)
            return None
        self = cls(counter)
        return self

    def _intervalOf(self, t: float) -> int:
        """
        Determine the number of intervals passed as of the given point in
        time.

        @param t: The specified time (from the start of the L{LoopingCall}) to
            be measured in intervals

        @return: The C{int} number of intervals which have passed as of the
            given point in time.
        """
        assert self.starttime is not None
        assert self.interval is not None
        elapsedTime = t - self.starttime
        intervalNum = int(elapsedTime / self.interval)
        return intervalNum

    def start(self, interval: float, now: bool=True) -> Deferred['LoopingCall']:
        """
        Start running function every interval seconds.

        @param interval: The number of seconds between calls.  May be
        less than one.  Precision will depend on the underlying
        platform, the available hardware, and the load on the system.

        @param now: If True, run this call right now.  Otherwise, wait
        until the interval has elapsed before beginning.

        @return: A Deferred whose callback will be invoked with
        C{self} when C{self.stop} is called, or whose errback will be
        invoked when the function raises an exception or returned a
        deferred that has its errback invoked.
        """
        assert not self.running, 'Tried to start an already running LoopingCall.'
        if interval < 0:
            raise ValueError('interval must be >= 0')
        self.running = True
        deferred = self._deferred = Deferred()
        self.starttime = self.clock.seconds()
        self.interval = interval
        self._runAtStart = now
        if now:
            self()
        else:
            self._scheduleFrom(self.starttime)
        return deferred

    def stop(self) -> None:
        """Stop running function."""
        assert self.running, 'Tried to stop a LoopingCall that was not running.'
        self.running = False
        if self.call is not None:
            self.call.cancel()
            self.call = None
            d, self._deferred = (self._deferred, None)
            assert d is not None
            d.callback(self)

    def reset(self) -> None:
        """
        Skip the next iteration and reset the timer.

        @since: 11.1
        """
        assert self.running, 'Tried to reset a LoopingCall that was not running.'
        if self.call is not None:
            self.call.cancel()
            self.call = None
            self.starttime = self.clock.seconds()
            self._scheduleFrom(self.starttime)

    def __call__(self) -> None:

        def cb(result: object) -> None:
            if self.running:
                self._scheduleFrom(self.clock.seconds())
            else:
                d, self._deferred = (self._deferred, None)
                assert d is not None
                d.callback(self)

        def eb(failure: Failure) -> None:
            self.running = False
            d, self._deferred = (self._deferred, None)
            assert d is not None
            d.errback(failure)
        self.call = None
        d = maybeDeferred(self.f, *self.a, **self.kw)
        d.addCallback(cb)
        d.addErrback(eb)

    def _scheduleFrom(self, when: float) -> None:
        """
        Schedule the next iteration of this looping call.

        @param when: The present time from whence the call is scheduled.
        """

        def howLong() -> float:
            if self.interval == 0:
                return 0
            assert self.starttime is not None
            runningFor = when - self.starttime
            assert self.interval is not None
            untilNextInterval = self.interval - runningFor % self.interval
            if when == when + untilNextInterval:
                return self.interval
            return untilNextInterval
        self.call = self.clock.callLater(howLong(), self)

    def __repr__(self) -> str:
        func = getattr(self.f, '__qualname__', None)
        if func is None:
            func = getattr(self.f, '__name__', None)
            if func is not None:
                imClass = getattr(self.f, 'im_class', None)
                if imClass is not None:
                    func = f'{imClass}.{func}'
        if func is None:
            func = reflect.safe_repr(self.f)
        return 'LoopingCall<{!r}>({}, *{}, **{})'.format(self.interval, func, reflect.safe_repr(self.a), reflect.safe_repr(self.kw))