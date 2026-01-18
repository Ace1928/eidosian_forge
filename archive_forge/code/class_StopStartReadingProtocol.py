import errno
import gc
import io
import os
import socket
from functools import wraps
from typing import Callable, ClassVar, List, Mapping, Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass, verifyObject
import attr
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import (
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import (
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.test.test_tcp import (
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
class StopStartReadingProtocol(Protocol):
    """
    Protocol that pauses and resumes the transport a few times
    """

    def connectionMade(self):
        self.data = b''
        self.pauseResumeProducing(3)

    def pauseResumeProducing(self, counter):
        """
        Toggle transport read state, then count down.
        """
        self.transport.pauseProducing()
        self.transport.resumeProducing()
        if counter:
            self.factory.reactor.callLater(0, self.pauseResumeProducing, counter - 1)
        else:
            self.factory.reactor.callLater(0, self.factory.ready.callback, self)

    def dataReceived(self, data):
        log.msg('got data', len(data))
        self.data += data
        if len(self.data) == 4 * 4096:
            self.factory.stop.callback(self.data)