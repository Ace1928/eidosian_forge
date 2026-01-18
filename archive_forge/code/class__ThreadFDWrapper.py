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
class _ThreadFDWrapper:
    """
    This wraps an event handler and translates notification in the helper
    L{Win32Reactor} thread into a notification in the primary reactor thread.

    @ivar _reactor: The primary reactor, the one to which event notification
        will be sent.

    @ivar _fd: The L{FileDescriptor} to which the event will be dispatched.

    @ivar _action: A C{str} giving the method of C{_fd} which handles the event.

    @ivar _logPrefix: The pre-fetched log prefix string for C{_fd}, so that
        C{_fd.logPrefix} does not need to be called in a non-main thread.
    """

    def __init__(self, reactor, fd, action, logPrefix):
        self._reactor = reactor
        self._fd = fd
        self._action = action
        self._logPrefix = logPrefix

    def logPrefix(self):
        """
        Return the original handler's log prefix, as it was given to
        C{__init__}.
        """
        return self._logPrefix

    def _execute(self):
        """
        Callback fired when the associated event is set.  Run the C{action}
        callback on the wrapped descriptor in the main reactor thread and raise
        or return whatever it raises or returns to cause this event handler to
        be removed from C{self._reactor} if appropriate.
        """
        return blockingCallFromThread(self._reactor, lambda: getattr(self._fd, self._action)())

    def connectionLost(self, reason):
        """
        Pass through to the wrapped descriptor, but in the main reactor thread
        instead of the helper C{Win32Reactor} thread.
        """
        self._reactor.callFromThread(self._fd.connectionLost, reason)