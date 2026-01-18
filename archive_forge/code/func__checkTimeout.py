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
def _checkTimeout(self, result: Union[str, Failure], name: str, lookupDeferred: Deferred[str]) -> None:
    try:
        userDeferred, cancelCall = self._runningQueries[lookupDeferred]
    except KeyError:
        pass
    else:
        del self._runningQueries[lookupDeferred]
        cancelCall.cancel()
        if isinstance(result, Failure):
            userDeferred.errback(self._fail(name, result.getErrorMessage()))
        else:
            userDeferred.callback(result)