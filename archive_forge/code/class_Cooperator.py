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
class Cooperator:
    """
    Cooperative task scheduler.

    A cooperative task is an iterator where each iteration represents an
    atomic unit of work.  When the iterator yields, it allows the
    L{Cooperator} to decide which of its tasks to execute next.  If the
    iterator yields a L{Deferred} then work will pause until the
    L{Deferred} fires and completes its callback chain.

    When a L{Cooperator} has more than one task, it distributes work between
    all tasks.

    There are two ways to add tasks to a L{Cooperator}, L{cooperate} and
    L{coiterate}.  L{cooperate} is the more useful of the two, as it returns a
    L{CooperativeTask}, which can be L{paused<CooperativeTask.pause>},
    L{resumed<CooperativeTask.resume>} and L{waited
    on<CooperativeTask.whenDone>}.  L{coiterate} has the same effect, but
    returns only a L{Deferred} that fires when the task is done.

    L{Cooperator} can be used for many things, including but not limited to:

      - running one or more computationally intensive tasks without blocking
      - limiting parallelism by running a subset of the total tasks
        simultaneously
      - doing one thing, waiting for a L{Deferred} to fire,
        doing the next thing, repeat (i.e. serializing a sequence of
        asynchronous tasks)

    Multiple L{Cooperator}s do not cooperate with each other, so for most
    cases you should use the L{global cooperator<task.cooperate>}.
    """

    def __init__(self, terminationPredicateFactory: Callable[[], Callable[[], bool]]=_Timer, scheduler: Callable[[Callable[[], None]], IDelayedCall]=_defaultScheduler, started: bool=True):
        """
        Create a scheduler-like object to which iterators may be added.

        @param terminationPredicateFactory: A no-argument callable which will
        be invoked at the beginning of each step and should return a
        no-argument callable which will return True when the step should be
        terminated.  The default factory is time-based and allows iterators to
        run for 1/100th of a second at a time.

        @param scheduler: A one-argument callable which takes a no-argument
        callable and should invoke it at some future point.  This will be used
        to schedule each step of this Cooperator.

        @param started: A boolean which indicates whether iterators should be
        stepped as soon as they are added, or if they will be queued up until
        L{Cooperator.start} is called.
        """
        self._tasks: List[CooperativeTask] = []
        self._metarator: Iterator[CooperativeTask] = iter(())
        self._terminationPredicateFactory = terminationPredicateFactory
        self._scheduler = scheduler
        self._delayedCall: Optional[IDelayedCall] = None
        self._stopped = False
        self._started = started

    def coiterate(self, iterator: Iterator[_TaskResultT], doneDeferred: Optional[Deferred[Iterator[_TaskResultT]]]=None) -> Deferred[Iterator[_TaskResultT]]:
        """
        Add an iterator to the list of iterators this L{Cooperator} is
        currently running.

        Equivalent to L{cooperate}, but returns a L{Deferred} that will
        be fired when the task is done.

        @param doneDeferred: If specified, this will be the Deferred used as
            the completion deferred.  It is suggested that you use the default,
            which creates a new Deferred for you.

        @return: a Deferred that will fire when the iterator finishes.
        """
        if doneDeferred is None:
            doneDeferred = Deferred()
        whenDone: Deferred[Iterator[_TaskResultT]] = CooperativeTask(iterator, self).whenDone()
        whenDone.chainDeferred(doneDeferred)
        return doneDeferred

    def cooperate(self, iterator: Iterator[_TaskResultT]) -> CooperativeTask:
        """
        Start running the given iterator as a long-running cooperative task, by
        calling next() on it as a periodic timed event.

        @param iterator: the iterator to invoke.

        @return: a L{CooperativeTask} object representing this task.
        """
        return CooperativeTask(iterator, self)

    def _addTask(self, task: CooperativeTask) -> None:
        """
        Add a L{CooperativeTask} object to this L{Cooperator}.
        """
        if self._stopped:
            self._tasks.append(task)
            task._completeWith(SchedulerStopped(), Failure(SchedulerStopped()))
        else:
            self._tasks.append(task)
            self._reschedule()

    def _removeTask(self, task: CooperativeTask) -> None:
        """
        Remove a L{CooperativeTask} from this L{Cooperator}.
        """
        self._tasks.remove(task)
        if not self._tasks and self._delayedCall:
            self._delayedCall.cancel()
            self._delayedCall = None

    def _tasksWhileNotStopped(self) -> Iterable[CooperativeTask]:
        """
        Yield all L{CooperativeTask} objects in a loop as long as this
        L{Cooperator}'s termination condition has not been met.
        """
        terminator = self._terminationPredicateFactory()
        while self._tasks:
            for t in self._metarator:
                yield t
                if terminator():
                    return
            self._metarator = iter(self._tasks)

    def _tick(self) -> None:
        """
        Run one scheduler tick.
        """
        self._delayedCall = None
        for taskObj in self._tasksWhileNotStopped():
            taskObj._oneWorkUnit()
        self._reschedule()
    _mustScheduleOnStart = False

    def _reschedule(self) -> None:
        if not self._started:
            self._mustScheduleOnStart = True
            return
        if self._delayedCall is None and self._tasks:
            self._delayedCall = self._scheduler(self._tick)

    def start(self) -> None:
        """
        Begin scheduling steps.
        """
        self._stopped = False
        self._started = True
        if self._mustScheduleOnStart:
            del self._mustScheduleOnStart
            self._reschedule()

    def stop(self) -> None:
        """
        Stop scheduling steps.  Errback the completion Deferreds of all
        iterators which have been added and forget about them.
        """
        self._stopped = True
        for taskObj in self._tasks:
            taskObj._completeWith(SchedulerStopped(), Failure(SchedulerStopped()))
        self._tasks = []
        if self._delayedCall is not None:
            self._delayedCall.cancel()
            self._delayedCall = None

    @property
    def running(self) -> bool:
        """
        Is this L{Cooperator} is currently running?

        @return: C{True} if the L{Cooperator} is running, C{False} otherwise.
        @rtype: C{bool}
        """
        return self._started and (not self._stopped)