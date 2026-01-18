import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
class PosixReactorBaseTests(WarningCheckerTestCase):
    """
    Tests for L{PosixReactorBase}.
    """

    def _checkWaker(self, reactor):
        self.assertIsInstance(reactor.waker, _Waker)
        self.assertIn(reactor.waker, reactor._internalReaders)
        self.assertIn(reactor.waker, reactor._readers)

    def test_wakerIsInternalReader(self):
        """
        When L{PosixReactorBase} is instantiated, it creates a waker and adds
        it to its internal readers set.
        """
        reactor = TrivialReactor()
        self._checkWaker(reactor)

    def test_removeAllSkipsInternalReaders(self):
        """
        Any L{IReadDescriptor}s in L{PosixReactorBase._internalReaders} are
        left alone by L{PosixReactorBase._removeAll}.
        """
        reactor = TrivialReactor()
        extra = object()
        reactor._internalReaders.add(extra)
        reactor.addReader(extra)
        reactor._removeAll(reactor._readers, reactor._writers)
        self._checkWaker(reactor)
        self.assertIn(extra, reactor._internalReaders)
        self.assertIn(extra, reactor._readers)

    def test_removeAllReturnsRemovedDescriptors(self):
        """
        L{PosixReactorBase._removeAll} returns a list of removed
        L{IReadDescriptor} and L{IWriteDescriptor} objects.
        """
        reactor = TrivialReactor()
        reader = object()
        writer = object()
        reactor.addReader(reader)
        reactor.addWriter(writer)
        removed = reactor._removeAll(reactor._readers, reactor._writers)
        self.assertEqual(set(removed), {reader, writer})
        self.assertNotIn(reader, reactor._readers)
        self.assertNotIn(writer, reactor._writers)