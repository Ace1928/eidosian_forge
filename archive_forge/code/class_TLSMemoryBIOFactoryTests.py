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
class TLSMemoryBIOFactoryTests(TestCase):
    """
    Ensure TLSMemoryBIOFactory logging acts correctly.
    """

    def test_quiet(self):
        """
        L{TLSMemoryBIOFactory.doStart} and L{TLSMemoryBIOFactory.doStop} do
        not log any messages.
        """
        contextFactory = ServerTLSContext()
        logs = []
        logger = logs.append
        log.addObserver(logger)
        self.addCleanup(log.removeObserver, logger)
        wrappedFactory = ServerFactory()
        wrappedFactory.doStart = lambda: None
        wrappedFactory.doStop = lambda: None
        factory = TLSMemoryBIOFactory(contextFactory, False, wrappedFactory)
        factory.doStart()
        factory.doStop()
        self.assertEqual(logs, [])

    def test_logPrefix(self):
        """
        L{TLSMemoryBIOFactory.logPrefix} amends the wrapped factory's log prefix
        with a short string (C{"TLS"}) indicating the wrapping, rather than its
        full class name.
        """
        contextFactory = ServerTLSContext()
        factory = TLSMemoryBIOFactory(contextFactory, False, ServerFactory())
        self.assertEqual('ServerFactory (TLS)', factory.logPrefix())

    def test_logPrefixFallback(self):
        """
        If the wrapped factory does not provide L{ILoggingContext},
        L{TLSMemoryBIOFactory.logPrefix} uses the wrapped factory's class name.
        """

        class NoFactory:
            pass
        contextFactory = ServerTLSContext()
        factory = TLSMemoryBIOFactory(contextFactory, False, NoFactory())
        self.assertEqual('NoFactory (TLS)', factory.logPrefix())