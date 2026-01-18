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
class _ThreePhaseEvent:
    """
    Collection of callables (with arguments) which can be invoked as a group in
    a particular order.

    This provides the underlying implementation for the reactor's system event
    triggers.  An instance of this class tracks triggers for all phases of a
    single type of event.

    @ivar before: A list of the before-phase triggers containing three-tuples
        of a callable, a tuple of positional arguments, and a dict of keyword
        arguments

    @ivar finishedBefore: A list of the before-phase triggers which have
        already been executed.  This is only populated in the C{'BEFORE'} state.

    @ivar during: A list of the during-phase triggers containing three-tuples
        of a callable, a tuple of positional arguments, and a dict of keyword
        arguments

    @ivar after: A list of the after-phase triggers containing three-tuples
        of a callable, a tuple of positional arguments, and a dict of keyword
        arguments

    @ivar state: A string indicating what is currently going on with this
        object.  One of C{'BASE'} (for when nothing in particular is happening;
        this is the initial value), C{'BEFORE'} (when the before-phase triggers
        are in the process of being executed).
    """

    def __init__(self) -> None:
        self.before: List[_ThreePhaseEventTrigger] = []
        self.during: List[_ThreePhaseEventTrigger] = []
        self.after: List[_ThreePhaseEventTrigger] = []
        self.state = 'BASE'

    def addTrigger(self, phase: str, callable: _ThreePhaseEventTriggerCallable, *args: object, **kwargs: object) -> _ThreePhaseEventTriggerHandle:
        """
        Add a trigger to the indicate phase.

        @param phase: One of C{'before'}, C{'during'}, or C{'after'}.

        @param callable: An object to be called when this event is triggered.
        @param args: Positional arguments to pass to C{callable}.
        @param kwargs: Keyword arguments to pass to C{callable}.

        @return: An opaque handle which may be passed to L{removeTrigger} to
            reverse the effects of calling this method.
        """
        if phase not in ('before', 'during', 'after'):
            raise KeyError('invalid phase')
        getattr(self, phase).append((callable, args, kwargs))
        return _ThreePhaseEventTriggerHandle((phase, callable, args, kwargs))

    def removeTrigger(self, handle: _ThreePhaseEventTriggerHandle) -> None:
        """
        Remove a previously added trigger callable.

        @param handle: An object previously returned by L{addTrigger}.  The
            trigger added by that call will be removed.

        @raise ValueError: If the trigger associated with C{handle} has already
            been removed or if C{handle} is not a valid handle.
        """
        getattr(self, 'removeTrigger_' + self.state)(handle)

    def removeTrigger_BASE(self, handle: _ThreePhaseEventTriggerHandle) -> None:
        """
        Just try to remove the trigger.

        @see: removeTrigger
        """
        try:
            phase, callable, args, kwargs = handle
        except (TypeError, ValueError):
            raise ValueError('invalid trigger handle')
        else:
            if phase not in ('before', 'during', 'after'):
                raise KeyError('invalid phase')
            getattr(self, phase).remove((callable, args, kwargs))

    def removeTrigger_BEFORE(self, handle: _ThreePhaseEventTriggerHandle) -> None:
        """
        Remove the trigger if it has yet to be executed, otherwise emit a
        warning that in the future an exception will be raised when removing an
        already-executed trigger.

        @see: removeTrigger
        """
        phase, callable, args, kwargs = handle
        if phase != 'before':
            return self.removeTrigger_BASE(handle)
        if (callable, args, kwargs) in self.finishedBefore:
            warnings.warn('Removing already-fired system event triggers will raise an exception in a future version of Twisted.', category=DeprecationWarning, stacklevel=3)
        else:
            self.removeTrigger_BASE(handle)

    def fireEvent(self) -> None:
        """
        Call the triggers added to this event.
        """
        self.state = 'BEFORE'
        self.finishedBefore = []
        beforeResults: List[Deferred[object]] = []
        while self.before:
            callable, args, kwargs = self.before.pop(0)
            self.finishedBefore.append((callable, args, kwargs))
            try:
                result = callable(*args, **kwargs)
            except BaseException:
                log.err()
            else:
                if isinstance(result, Deferred):
                    beforeResults.append(result)
        DeferredList(beforeResults).addCallback(self._continueFiring)

    def _continueFiring(self, ignored: object) -> None:
        """
        Call the during and after phase triggers for this event.
        """
        self.state = 'BASE'
        self.finishedBefore = []
        for phase in (self.during, self.after):
            while phase:
                callable, args, kwargs = phase.pop(0)
                try:
                    callable(*args, **kwargs)
                except BaseException:
                    log.err()