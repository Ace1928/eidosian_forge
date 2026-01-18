from __future__ import annotations
from random import randrange
from typing import Any, Callable, TextIO, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import interfaces, utils
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
class _Janitor:
    """
    The guy that cleans up after you.

    @ivar test: The L{TestCase} to report errors about.
    @ivar result: The L{IReporter} to report errors to.
    @ivar reactor: The reactor to use. If None, the global reactor
        will be used.
    """

    def __init__(self, test, result, reactor=None):
        """
        @param test: See L{_Janitor.test}.
        @param result: See L{_Janitor.result}.
        @param reactor: See L{_Janitor.reactor}.
        """
        self.test = test
        self.result = result
        self.reactor = reactor

    def postCaseCleanup(self):
        """
        Called by L{unittest.TestCase} after a test to catch any logged errors
        or pending L{DelayedCall<twisted.internet.base.DelayedCall>}s.
        """
        calls = self._cleanPending()
        if calls:
            aggregate = DirtyReactorAggregateError(calls)
            self.result.addError(self.test, Failure(aggregate))
            return False
        return True

    def postClassCleanup(self):
        """
        Called by L{unittest.TestCase} after the last test in a C{TestCase}
        subclass. Ensures the reactor is clean by murdering the threadpool,
        catching any pending
        L{DelayedCall<twisted.internet.base.DelayedCall>}s, open sockets etc.
        """
        selectables = self._cleanReactor()
        calls = self._cleanPending()
        if selectables or calls:
            aggregate = DirtyReactorAggregateError(calls, selectables)
            self.result.addError(self.test, Failure(aggregate))
        self._cleanThreads()

    def _getReactor(self):
        """
        Get either the passed-in reactor or the global reactor.
        """
        if self.reactor is not None:
            reactor = self.reactor
        else:
            from twisted.internet import reactor
        return reactor

    def _cleanPending(self):
        """
        Cancel all pending calls and return their string representations.
        """
        reactor = self._getReactor()
        reactor.iterate(0)
        reactor.iterate(0)
        delayedCallStrings = []
        for p in reactor.getDelayedCalls():
            if p.active():
                delayedString = str(p)
                p.cancel()
            else:
                print('WEIRDNESS! pending timed call not active!')
            delayedCallStrings.append(delayedString)
        return delayedCallStrings
    _cleanPending = utils.suppressWarnings(_cleanPending, (('ignore',), {'category': DeprecationWarning, 'message': 'reactor\\.iterate cannot be used.*'}))

    def _cleanThreads(self):
        reactor = self._getReactor()
        if interfaces.IReactorThreads.providedBy(reactor):
            if reactor.threadpool is not None:
                reactor._stopThreadPool()

    def _cleanReactor(self):
        """
        Remove all selectables from the reactor, kill any of them that were
        processes, and return their string representation.
        """
        reactor = self._getReactor()
        selectableStrings = []
        for sel in reactor.removeAll():
            if interfaces.IProcessTransport.providedBy(sel):
                sel.signalProcess('KILL')
            selectableStrings.append(repr(sel))
        return selectableStrings