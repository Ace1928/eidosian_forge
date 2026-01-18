import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
class BadClient(protocol.DatagramProtocol):
    """
    A DatagramProtocol which always raises an exception from datagramReceived.
    Used to test error handling behavior in the reactor for that method.
    """
    d = None

    def setDeferred(self, d):
        """
        Set the Deferred which will be called back when datagramReceived is
        called.
        """
        self.d = d

    def datagramReceived(self, bytes, addr):
        if self.d is not None:
            d, self.d = (self.d, None)
            d.callback(bytes)
        raise BadClientError('Application code is very buggy!')