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
def _stopThreadPool(self) -> None:
    """
            Stop the reactor threadpool.  This method is only valid if there
            is currently a threadpool (created by L{_initThreadPool}).  It
            is not intended to be called directly; instead, it will be
            called by a shutdown trigger created in L{_initThreadPool}.
            """
    triggers = [self._threadpoolStartupID, self.threadpoolShutdownID]
    for trigger in filter(None, triggers):
        try:
            self.removeSystemEventTrigger(trigger)
        except ValueError:
            pass
    self._threadpoolStartupID = None
    self.threadpoolShutdownID = None
    assert self.threadpool is not None
    self.threadpool.stop()
    self.threadpool = None