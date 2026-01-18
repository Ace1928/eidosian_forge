import errno
import os
import socket
from unittest import skipIf
from twisted.internet import interfaces, reactor
from twisted.internet.defer import gatherResults, maybeDeferred
from twisted.internet.protocol import Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.python import log
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def _acceptFailureTest(self, socketErrorNumber):
    """
        Test behavior in the face of an exception from C{accept(2)}.

        On any exception which indicates the platform is unable or unwilling
        to allocate further resources to us, the existing port should remain
        listening, a message should be logged, and the exception should not
        propagate outward from doRead.

        @param socketErrorNumber: The errno to simulate from accept.
        """

    class FakeSocket:
        """
            Pretend to be a socket in an overloaded system.
            """

        def accept(self):
            raise OSError(socketErrorNumber, os.strerror(socketErrorNumber))
    factory = ServerFactory()
    port = self.port(0, factory, interface='127.0.0.1')
    self.patch(port, 'socket', FakeSocket())
    port.doRead()
    expectedFormat = 'Could not accept new connection ({acceptError})'
    expectedErrorCode = errno.errorcode[socketErrorNumber]
    matchingMessages = [msg.get('log_format') == expectedFormat and msg.get('acceptError') == expectedErrorCode for msg in self.messages]
    self.assertGreater(len(matchingMessages), 0, 'Log event for failed accept not found in %r' % (self.messages,))