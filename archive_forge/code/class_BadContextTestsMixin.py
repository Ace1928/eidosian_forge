from typing import Optional, Sequence, Type
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.endpoints import (
from twisted.internet.error import ConnectionClosed
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
class BadContextTestsMixin:
    """
    Mixin for L{ReactorBuilder} subclasses which defines a helper for testing
    the handling of broken context factories.
    """

    def _testBadContext(self, useIt):
        """
        Assert that the exception raised by a broken context factory's
        C{getContext} method is raised by some reactor method.  If it is not, an
        exception will be raised to fail the test.

        @param useIt: A two-argument callable which will be called with a
            reactor and a broken context factory and which is expected to raise
            the same exception as the broken context factory's C{getContext}
            method.
        """
        reactor = self.buildReactor()
        exc = self.assertRaises(ValueError, useIt, reactor, BrokenContextFactory())
        self.assertEqual(BrokenContextFactory.message, str(exc))