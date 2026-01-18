from __future__ import annotations
import gc
from typing import Union
from zope.interface import Interface, directlyProvides, implementer
from zope.interface.verify import verifyObject
from hypothesis import given, strategies as st
from twisted.internet import reactor
from twisted.internet.task import Clock, deferLater
from twisted.python.compat import iterbytes
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol, ServerFactory
from twisted.internet.task import TaskStopped
from twisted.internet.testing import NonStreamingProducer, StringTransport
from twisted.protocols.loopback import collapsingPumpPolicy, loopbackAsync
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SynchronousTestCase, TestCase
class HistoryStringTransport(StringTransport):

    def __init__(self):
        StringTransport.__init__(self)
        self.producerHistory = []

    def pauseProducing(self):
        self.producerHistory.append('pause')
        StringTransport.pauseProducing(self)

    def resumeProducing(self):
        self.producerHistory.append('resume')
        StringTransport.resumeProducing(self)

    def stopProducing(self):
        self.producerHistory.append('stop')
        StringTransport.stopProducing(self)