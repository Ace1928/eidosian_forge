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