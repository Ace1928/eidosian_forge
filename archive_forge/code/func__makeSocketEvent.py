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
def _makeSocketEvent(self, fd, action, why):
    """
        Make a win32 event object for a socket.
        """
    event = CreateEvent(None, 0, 0, None)
    WSAEventSelect(fd, event, why)
    self._events[event] = (fd, action)
    return event