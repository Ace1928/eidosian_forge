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
def _runAction(self, action, fd):
    try:
        closed = getattr(fd, action)()
    except BaseException:
        closed = sys.exc_info()[1]
        log.deferr()
    if closed:
        self._disconnectSelectable(fd, closed, action == 'doRead')