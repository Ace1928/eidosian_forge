import socket
from queue import Queue
from typing import Callable
from unittest import skipIf
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet._resolver import FirstOneWins
from twisted.internet.base import DelayedCall, ReactorBase, ThreadedResolver
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import IReactorThreads, IReactorTime, IResolverSimple
from twisted.internet.task import Clock
from twisted.python.threadpool import ThreadPool
from twisted.trial.unittest import SkipTest, TestCase
def _getDelayedCallAt(self, time):
    """
        Get a L{DelayedCall} instance at a given C{time}.

        @param time: The absolute time at which the returned L{DelayedCall}
            will be scheduled.
        """

    def noop(call):
        pass
    return DelayedCall(time, lambda: None, (), {}, noop, noop, None)