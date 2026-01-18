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
def _runWrite(self, fd):
    closed = 0
    try:
        closed = fd.doWrite()
    except BaseException:
        closed = sys.exc_info()[1]
        log.deferr()
    if closed:
        self.removeReader(fd)
        self.removeWriter(fd)
        try:
            fd.connectionLost(failure.Failure(closed))
        except BaseException:
            log.deferr()
    elif closed is None:
        return 1