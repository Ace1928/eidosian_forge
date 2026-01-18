from random import random as _goodEnoughRandom
from typing import List
from automat import MethodicalMachine
from twisted.application import service
from twisted.internet import task
from twisted.internet.defer import (
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
class TimerService(_VolatileDataService):
    """
    Service to periodically call a function

    Every C{step} seconds call the given function with the given arguments.
    The service starts the calls when it starts, and cancels them
    when it stops.

    @ivar clock: Source of time. This defaults to L{None} which is
        causes L{twisted.internet.reactor} to be used.
        Feel free to set this to something else, but it probably ought to be
        set *before* calling L{startService}.
    @type clock: L{IReactorTime<twisted.internet.interfaces.IReactorTime>}

    @ivar call: Function and arguments to call periodically.
    @type call: L{tuple} of C{(callable, args, kwargs)}
    """
    volatile = ['_loop', '_loopFinished']

    def __init__(self, step, callable, *args, **kwargs):
        """
        @param step: The number of seconds between calls.
        @type step: L{float}

        @param callable: Function to call
        @type callable: L{callable}

        @param args: Positional arguments to pass to function
        @param kwargs: Keyword arguments to pass to function
        """
        self.step = step
        self.call = (callable, args, kwargs)
        self.clock = None

    def startService(self):
        service.Service.startService(self)
        callable, args, kwargs = self.call
        self._loop = task.LoopingCall(callable, *args, **kwargs)
        self._loop.clock = _maybeGlobalReactor(self.clock)
        self._loopFinished = self._loop.start(self.step, now=True)
        self._loopFinished.addErrback(self._failed)

    def _failed(self, why):
        self._loop.running = False
        log.err(why)

    def stopService(self):
        """
        Stop the service.

        @rtype: L{Deferred<defer.Deferred>}
        @return: a L{Deferred<defer.Deferred>} which is fired when the
            currently running call (if any) is finished.
        """
        if self._loop.running:
            self._loop.stop()
        self._loopFinished.addCallback(lambda _: service.Service.stopService(self))
        return self._loopFinished