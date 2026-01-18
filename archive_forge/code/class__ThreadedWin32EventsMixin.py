import sys
import time
from threading import Thread
from weakref import WeakKeyDictionary
from zope.interface import implementer
from win32file import FD_ACCEPT, FD_CLOSE, FD_CONNECT, FD_READ, WSAEventSelect
import win32gui  # type: ignore[import-untyped]
from win32event import (
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet, IReactorWin32Events
from twisted.internet.threads import blockingCallFromThread
from twisted.python import failure, log, threadable
@implementer(IReactorWin32Events)
class _ThreadedWin32EventsMixin:
    """
    This mixin implements L{IReactorWin32Events} for another reactor by running
    a L{Win32Reactor} in a separate thread and dispatching work to it.

    @ivar _reactor: The L{Win32Reactor} running in the other thread.  This is
        L{None} until it is actually needed.

    @ivar _reactorThread: The L{threading.Thread} which is running the
        L{Win32Reactor}.  This is L{None} until it is actually needed.
    """
    _reactor = None
    _reactorThread = None

    def _unmakeHelperReactor(self):
        """
        Stop and discard the reactor started by C{_makeHelperReactor}.
        """
        self._reactor.callFromThread(self._reactor.stop)
        self._reactor = None

    def _makeHelperReactor(self):
        """
        Create and (in a new thread) start a L{Win32Reactor} instance to use for
        the implementation of L{IReactorWin32Events}.
        """
        self._reactor = Win32Reactor()
        self._reactor._registerAsIOThread = False
        self._reactorThread = Thread(target=self._reactor.run, args=(False,))
        self.addSystemEventTrigger('after', 'shutdown', self._unmakeHelperReactor)
        self._reactorThread.start()

    def addEvent(self, event, fd, action):
        """
        @see: L{IReactorWin32Events}
        """
        if self._reactor is None:
            self._makeHelperReactor()
        self._reactor.callFromThread(self._reactor.addEvent, event, _ThreadFDWrapper(self, fd, action, fd.logPrefix()), '_execute')

    def removeEvent(self, event):
        """
        @see: L{IReactorWin32Events}
        """
        self._reactor.callFromThread(self._reactor.removeEvent, event)