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