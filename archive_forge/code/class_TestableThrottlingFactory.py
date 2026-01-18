import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
class TestableThrottlingFactory(policies.ThrottlingFactory):
    """
    L{policies.ThrottlingFactory} using a L{task.Clock} for tests.
    """

    def __init__(self, clock, *args, **kwargs):
        """
        @param clock: object providing a callLater method that can be used
            for tests.
        @type clock: C{task.Clock} or alike.
        """
        policies.ThrottlingFactory.__init__(self, *args, **kwargs)
        self.clock = clock

    def callLater(self, period, func):
        """
        Forward to the testable clock.
        """
        return self.clock.callLater(period, func)