import builtins
import socket  # needed only for sync-dns
import warnings
from abc import ABC, abstractmethod
from heapq import heapify, heappop, heappush
from traceback import format_stack
from types import FrameType
from typing import (
from zope.interface import classImplements, implementer
from twisted.internet import abstract, defer, error, fdesc, main, threads
from twisted.internet._resolver import (
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory
from twisted.python import log, reflect
from twisted.python.failure import Failure
from twisted.python.runtime import platform, seconds as runtimeSeconds
from ._signals import SignalHandling, _WithoutSignalHandling, _WithSignalHandling
from twisted.python import threadable
@implementer(IReactorCore, IReactorTime, _ISupportsExitSignalCapturing)
class ReactorBase(PluggableResolverMixin):
    """
    Default base class for Reactors.

    @ivar _stopped: A flag which is true between paired calls to C{reactor.run}
        and C{reactor.stop}.  This should be replaced with an explicit state
        machine.
    @ivar _justStopped: A flag which is true between the time C{reactor.stop}
        is called and the time the shutdown system event is fired.  This is
        used to determine whether that event should be fired after each
        iteration through the mainloop.  This should be replaced with an
        explicit state machine.
    @ivar _started: A flag which is true from the time C{reactor.run} is called
        until the time C{reactor.run} returns.  This is used to prevent calls
        to C{reactor.run} on a running reactor.  This should be replaced with
        an explicit state machine.
    @ivar running: See L{IReactorCore.running}
    @ivar _registerAsIOThread: A flag controlling whether the reactor will
        register the thread it is running in as the I/O thread when it starts.
        If C{True}, registration will be done, otherwise it will not be.
    @ivar _exitSignal: See L{_ISupportsExitSignalCapturing._exitSignal}

    @ivar _installSignalHandlers: A flag which indicates whether any signal
        handlers will be installed during startup.  This includes handlers for
        SIGCHLD to monitor child processes, and SIGINT, SIGTERM, and SIGBREAK

    @ivar _signals: An object which knows how to install and uninstall the
        reactor's signal-handling behavior.
    """
    _registerAsIOThread = True
    _stopped = True
    installed = False
    usingThreads = False
    _exitSignal = None
    _signals: SignalHandling = _WithoutSignalHandling()
    __name__ = 'twisted.internet.reactor'

    def __init__(self) -> None:
        super().__init__()
        self.threadCallQueue: List[_ThreadCall] = []
        self._eventTriggers: Dict[str, _ThreePhaseEvent] = {}
        self._pendingTimedCalls: List[DelayedCall] = []
        self._newTimedCalls: List[DelayedCall] = []
        self._cancellations = 0
        self.running = False
        self._started = False
        self._justStopped = False
        self._startedBefore = False
        self._internalReaders: Set[Any] = set()
        self.waker: Any = None
        self.addSystemEventTrigger('during', 'startup', self._reallyStartRunning)
        self.addSystemEventTrigger('during', 'shutdown', self.crash)
        self.addSystemEventTrigger('during', 'shutdown', self.disconnectAll)
        if platform.supportsThreads():
            self._initThreads()
        self.installWaker()
    _installSignalHandlers: bool = False

    def _makeSignalHandling(self, installSignalHandlers: bool) -> SignalHandling:
        """
        Get an appropriate signal handling object.

        @param installSignalHandlers: Indicate whether to even try to do any
            signal handling.  If C{False} then the result will be a no-op
            implementation.
        """
        if installSignalHandlers:
            return self._signalsFactory()
        return _WithoutSignalHandling()

    def _signalsFactory(self) -> SignalHandling:
        """
        Get a signal handling object that implements the basic behavior of
        stopping the reactor on SIGINT, SIGBREAK, and SIGTERM.
        """
        return _WithSignalHandling(self.sigInt, self.sigBreak, self.sigTerm)

    def _addInternalReader(self, reader: IReadDescriptor) -> None:
        """
        Add a read descriptor which is part of the implementation of the
        reactor itself.

        The read descriptor will not be removed by L{IReactorFDSet.removeAll}.
        """
        self._internalReaders.add(reader)
        self.addReader(reader)

    def _removeInternalReader(self, reader: IReadDescriptor) -> None:
        """
        Remove a read descriptor which is part of the implementation of the
        reactor itself.
        """
        self._internalReaders.remove(reader)
        self.removeReader(reader)

    def run(self, installSignalHandlers: bool=True) -> None:
        self.startRunning(installSignalHandlers=installSignalHandlers)
        try:
            self.mainLoop()
        finally:
            self._signals.uninstall()

    def mainLoop(self) -> None:
        while self._started:
            try:
                while self._started:
                    self.runUntilCurrent()
                    t2 = self.timeout()
                    t = self.running and t2
                    self.doIteration(t)
            except BaseException:
                log.msg('Unexpected error in main loop.')
                log.err()
            else:
                log.msg('Main loop terminated.')
    _lock = None

    def installWaker(self) -> None:
        raise NotImplementedError(reflect.qual(self.__class__) + ' did not implement installWaker')

    def wakeUp(self) -> None:
        """
        Wake up the event loop.
        """
        if self.waker:
            self.waker.wakeUp()

    def doIteration(self, delay: Optional[float]) -> None:
        """
        Do one iteration over the readers and writers which have been added.
        """
        raise NotImplementedError(reflect.qual(self.__class__) + ' did not implement doIteration')

    def addReader(self, reader: IReadDescriptor) -> None:
        raise NotImplementedError(reflect.qual(self.__class__) + ' did not implement addReader')

    def addWriter(self, writer: IWriteDescriptor) -> None:
        raise NotImplementedError(reflect.qual(self.__class__) + ' did not implement addWriter')

    def removeReader(self, reader: IReadDescriptor) -> None:
        raise NotImplementedError(reflect.qual(self.__class__) + ' did not implement removeReader')

    def removeWriter(self, writer: IWriteDescriptor) -> None:
        raise NotImplementedError(reflect.qual(self.__class__) + ' did not implement removeWriter')

    def removeAll(self) -> List[Union[IReadDescriptor, IWriteDescriptor]]:
        raise NotImplementedError(reflect.qual(self.__class__) + ' did not implement removeAll')

    def getReaders(self) -> List[IReadDescriptor]:
        raise NotImplementedError(reflect.qual(self.__class__) + ' did not implement getReaders')

    def getWriters(self) -> List[IWriteDescriptor]:
        raise NotImplementedError(reflect.qual(self.__class__) + ' did not implement getWriters')

    def resolve(self, name: str, timeout: Sequence[int]=(1, 3, 11, 45)) -> Deferred[str]:
        """
        Return a Deferred that will resolve a hostname."""
        if not name:
            return defer.succeed('0.0.0.0')
        if abstract.isIPAddress(name):
            return defer.succeed(name)
        return self.resolver.getHostByName(name, timeout)

    def stop(self) -> None:
        """
        See twisted.internet.interfaces.IReactorCore.stop.
        """
        if self._stopped:
            raise error.ReactorNotRunning("Can't stop reactor that isn't running.")
        self._stopped = True
        self._justStopped = True
        self._startedBefore = True

    def crash(self) -> None:
        """
        See twisted.internet.interfaces.IReactorCore.crash.

        Reset reactor state tracking attributes and re-initialize certain
        state-transition helpers which were set up in C{__init__} but later
        destroyed (through use).
        """
        self._started = False
        self.running = False
        self.addSystemEventTrigger('during', 'startup', self._reallyStartRunning)

    def sigInt(self, number: int, frame: Optional[FrameType]=None) -> None:
        """
        Handle a SIGINT interrupt.

        @param number: See handler specification in L{signal.signal}
        @param frame: See handler specification in L{signal.signal}
        """
        log.msg('Received SIGINT, shutting down.')
        self.callFromThread(self.stop)
        self._exitSignal = number

    def sigBreak(self, number: int, frame: Optional[FrameType]=None) -> None:
        """
        Handle a SIGBREAK interrupt.

        @param number: See handler specification in L{signal.signal}
        @param frame: See handler specification in L{signal.signal}
        """
        log.msg('Received SIGBREAK, shutting down.')
        self.callFromThread(self.stop)
        self._exitSignal = number

    def sigTerm(self, number: int, frame: Optional[FrameType]=None) -> None:
        """
        Handle a SIGTERM interrupt.

        @param number: See handler specification in L{signal.signal}
        @param frame: See handler specification in L{signal.signal}
        """
        log.msg('Received SIGTERM, shutting down.')
        self.callFromThread(self.stop)
        self._exitSignal = number

    def disconnectAll(self) -> None:
        """Disconnect every reader, and writer in the system."""
        selectables = self.removeAll()
        for reader in selectables:
            log.callWithLogger(reader, reader.connectionLost, Failure(main.CONNECTION_LOST))

    def iterate(self, delay: float=0.0) -> None:
        """
        See twisted.internet.interfaces.IReactorCore.iterate.
        """
        self.runUntilCurrent()
        self.doIteration(delay)

    def fireSystemEvent(self, eventType: str) -> None:
        """
        See twisted.internet.interfaces.IReactorCore.fireSystemEvent.
        """
        event = self._eventTriggers.get(eventType)
        if event is not None:
            event.fireEvent()

    def addSystemEventTrigger(self, phase: str, eventType: str, callable: Callable[..., Any], *args: object, **kwargs: object) -> _SystemEventID:
        """
        See twisted.internet.interfaces.IReactorCore.addSystemEventTrigger.
        """
        assert builtins.callable(callable), f'{callable} is not callable'
        if eventType not in self._eventTriggers:
            self._eventTriggers[eventType] = _ThreePhaseEvent()
        return _SystemEventID((eventType, self._eventTriggers[eventType].addTrigger(phase, callable, *args, **kwargs)))

    def removeSystemEventTrigger(self, triggerID: _SystemEventID) -> None:
        """
        See twisted.internet.interfaces.IReactorCore.removeSystemEventTrigger.
        """
        eventType, handle = triggerID
        self._eventTriggers[eventType].removeTrigger(handle)

    def callWhenRunning(self, callable: Callable[..., Any], *args: object, **kwargs: object) -> Optional[_SystemEventID]:
        """
        See twisted.internet.interfaces.IReactorCore.callWhenRunning.
        """
        if self.running:
            callable(*args, **kwargs)
            return None
        else:
            return self.addSystemEventTrigger('after', 'startup', callable, *args, **kwargs)

    def startRunning(self, installSignalHandlers: bool=True) -> None:
        """
        Method called when reactor starts: do some initialization and fire
        startup events.

        Don't call this directly, call reactor.run() instead: it should take
        care of calling this.

        This method is somewhat misnamed.  The reactor will not necessarily be
        in the running state by the time this method returns.  The only
        guarantee is that it will be on its way to the running state.

        @param installSignalHandlers: A flag which, if set, indicates that
            handlers for a number of (implementation-defined) signals should be
            installed during startup.
        """
        if self._started:
            raise error.ReactorAlreadyRunning()
        if self._startedBefore:
            raise error.ReactorNotRestartable()
        self._signals.uninstall()
        self._installSignalHandlers = installSignalHandlers
        self._signals = self._makeSignalHandling(installSignalHandlers)
        self._started = True
        self._stopped = False
        if self._registerAsIOThread:
            threadable.registerAsIOThread()
        self.fireSystemEvent('startup')

    def _reallyStartRunning(self) -> None:
        """
        Method called to transition to the running state.  This should happen
        in the I{during startup} event trigger phase.
        """
        self.running = True
        if self._installSignalHandlers:
            self._signals.install()
    seconds = staticmethod(runtimeSeconds)

    def callLater(self, delay: float, callable: Callable[..., Any], *args: object, **kw: object) -> DelayedCall:
        """
        See twisted.internet.interfaces.IReactorTime.callLater.
        """
        assert builtins.callable(callable), f'{callable} is not callable'
        assert delay >= 0, f'{delay} is not greater than or equal to 0 seconds'
        delayedCall = DelayedCall(self.seconds() + delay, callable, args, kw, self._cancelCallLater, self._moveCallLaterSooner, seconds=self.seconds)
        self._newTimedCalls.append(delayedCall)
        return delayedCall

    def _moveCallLaterSooner(self, delayedCall: DelayedCall) -> None:
        heap = self._pendingTimedCalls
        try:
            pos = heap.index(delayedCall)
            elt = heap[pos]
            while pos != 0:
                parent = (pos - 1) // 2
                if heap[parent] <= elt:
                    break
                heap[pos] = heap[parent]
                pos = parent
            heap[pos] = elt
        except ValueError:
            pass

    def _cancelCallLater(self, delayedCall: DelayedCall) -> None:
        self._cancellations += 1

    def getDelayedCalls(self) -> Sequence[IDelayedCall]:
        """
        See L{twisted.internet.interfaces.IReactorTime.getDelayedCalls}
        """
        return [x for x in self._pendingTimedCalls + self._newTimedCalls if not x.cancelled]

    def _insertNewDelayedCalls(self) -> None:
        for call in self._newTimedCalls:
            if call.cancelled:
                self._cancellations -= 1
            else:
                call.activate_delay()
                heappush(self._pendingTimedCalls, call)
        self._newTimedCalls = []

    def timeout(self) -> Optional[float]:
        """
        Determine the longest time the reactor may sleep (waiting on I/O
        notification, perhaps) before it must wake up to service a time-related
        event.

        @return: The maximum number of seconds the reactor may sleep.
        """
        self._insertNewDelayedCalls()
        if not self._pendingTimedCalls:
            return None
        delay = self._pendingTimedCalls[0].time - self.seconds()
        longest = 2147483
        return max(0, min(longest, delay))

    def runUntilCurrent(self) -> None:
        """
        Run all pending timed calls.
        """
        if self.threadCallQueue:
            count = 0
            total = len(self.threadCallQueue)
            for f, a, kw in self.threadCallQueue:
                try:
                    f(*a, **kw)
                except BaseException:
                    log.err()
                count += 1
                if count == total:
                    break
            del self.threadCallQueue[:count]
            if self.threadCallQueue:
                self.wakeUp()
        self._insertNewDelayedCalls()
        now = self.seconds()
        while self._pendingTimedCalls and self._pendingTimedCalls[0].time <= now:
            call = heappop(self._pendingTimedCalls)
            if call.cancelled:
                self._cancellations -= 1
                continue
            if call.delayed_time > 0.0:
                call.activate_delay()
                heappush(self._pendingTimedCalls, call)
                continue
            try:
                call.called = 1
                call.func(*call.args, **call.kw)
            except BaseException:
                log.err()
                if call.creator is not None:
                    e = '\n'
                    e += ' C: previous exception occurred in ' + 'a DelayedCall created here:\n'
                    e += ' C:'
                    e += ''.join(call.creator).rstrip().replace('\n', '\n C:')
                    e += '\n'
                    log.msg(e)
        if self._cancellations > 50 and self._cancellations > len(self._pendingTimedCalls) >> 1:
            self._cancellations = 0
            self._pendingTimedCalls = [x for x in self._pendingTimedCalls if not x.cancelled]
            heapify(self._pendingTimedCalls)
        if self._justStopped:
            self._justStopped = False
            self.fireSystemEvent('shutdown')
    if platform.supportsThreads():
        assert ThreadPool is not None
        threadpool = None
        _threadpoolStartupID = None
        threadpoolShutdownID = None

        def _initThreads(self) -> None:
            self.installNameResolver(_GAIResolver(cast(IReactorThreads, self), self.getThreadPool))
            self.usingThreads = True

        def callFromThread(self, f: Callable[..., Any], *args: object, **kwargs: object) -> None:
            """
            See
            L{twisted.internet.interfaces.IReactorFromThreads.callFromThread}.
            """
            assert callable(f), f'{f} is not callable'
            self.threadCallQueue.append((f, args, kwargs))
            self.wakeUp()

        def _initThreadPool(self) -> None:
            """
            Create the threadpool accessible with callFromThread.
            """
            self.threadpool = ThreadPool(0, 10, 'twisted.internet.reactor')
            self._threadpoolStartupID = self.callWhenRunning(self.threadpool.start)
            self.threadpoolShutdownID = self.addSystemEventTrigger('during', 'shutdown', self._stopThreadPool)

        def _uninstallHandler(self) -> None:
            self._signals.uninstall()

        def _stopThreadPool(self) -> None:
            """
            Stop the reactor threadpool.  This method is only valid if there
            is currently a threadpool (created by L{_initThreadPool}).  It
            is not intended to be called directly; instead, it will be
            called by a shutdown trigger created in L{_initThreadPool}.
            """
            triggers = [self._threadpoolStartupID, self.threadpoolShutdownID]
            for trigger in filter(None, triggers):
                try:
                    self.removeSystemEventTrigger(trigger)
                except ValueError:
                    pass
            self._threadpoolStartupID = None
            self.threadpoolShutdownID = None
            assert self.threadpool is not None
            self.threadpool.stop()
            self.threadpool = None

        def getThreadPool(self) -> ThreadPool:
            """
            See L{twisted.internet.interfaces.IReactorThreads.getThreadPool}.
            """
            if self.threadpool is None:
                self._initThreadPool()
                assert self.threadpool is not None
            return self.threadpool

        def callInThread(self, _callable: Callable[..., Any], *args: object, **kwargs: object) -> None:
            """
            See L{twisted.internet.interfaces.IReactorInThreads.callInThread}.
            """
            self.getThreadPool().callInThread(_callable, *args, **kwargs)

        def suggestThreadPoolSize(self, size: int) -> None:
            """
            See L{twisted.internet.interfaces.IReactorThreads.suggestThreadPoolSize}.
            """
            self.getThreadPool().adjustPoolsize(maxthreads=size)
    else:

        def callFromThread(self, f: Callable[..., Any], *args: object, **kwargs: object) -> None:
            assert callable(f), f'{f} is not callable'
            self.threadCallQueue.append((f, args, kwargs))