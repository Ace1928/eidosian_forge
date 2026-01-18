from __future__ import annotations
from io import BytesIO
from socket import AF_INET, AF_INET6
from typing import Callable, Iterator, Sequence, overload
from zope.interface import implementedBy, implementer
from zope.interface.verify import verifyClass
from typing_extensions import ParamSpec, Self
from twisted.internet import address, error, protocol, task
from twisted.internet.abstract import _dataMustBeBytes, isIPv6Address
from twisted.internet.address import IPv4Address, IPv6Address, UNIXAddress
from twisted.internet.defer import Deferred
from twisted.internet.error import UnsupportedAddressFamily
from twisted.internet.interfaces import (
from twisted.internet.task import Clock
from twisted.logger import ILogObserver, LogEvent, LogPublisher
from twisted.protocols import basic
from twisted.python import failure
from twisted.trial.unittest import TestCase
class NonStreamingProducer:
    """
    A pull producer which writes 10 times only.
    """
    counter = 0
    stopped = False

    def __init__(self, consumer):
        self.consumer = consumer
        self.result = Deferred()

    def resumeProducing(self):
        """
        Write the counter value once.
        """
        if self.consumer is None or self.counter >= 10:
            raise RuntimeError('BUG: resume after unregister/stop.')
        else:
            self.consumer.write(b'%d' % (self.counter,))
            self.counter += 1
            if self.counter == 10:
                self.consumer.unregisterProducer()
                self._done()

    def pauseProducing(self):
        """
        An implementation of C{IPushProducer.pauseProducing}. This should never
        be called on a pull producer, so this just raises an error.
        """
        raise RuntimeError('BUG: pause should never be called.')

    def _done(self):
        """
        Fire a L{Deferred} so that users can wait for this to complete.
        """
        self.consumer = None
        d = self.result
        del self.result
        d.callback(None)

    def stopProducing(self):
        """
        Stop all production.
        """
        self.stopped = True
        self._done()