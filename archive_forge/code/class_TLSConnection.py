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
class TLSConnection:

    def __init__(self):
        self.l = []

    def send(self, data):
        if not self.l:
            data = data[:-1]
        if len(self.l) == 1:
            self.l.append('paused')
            raise WantReadError()
        self.l.append(data)
        return len(data)

    def set_connect_state(self):
        pass

    def do_handshake(self):
        pass

    def bio_write(self, data):
        pass

    def bio_read(self, size):
        return b'X'

    def recv(self, size):
        raise WantReadError()