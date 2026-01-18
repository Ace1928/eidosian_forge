import os
import signal
import struct
import sys
from unittest import skipIf
from zope.interface import implementer
from twisted.internet import defer, error, protocol
from twisted.internet.address import IPv4Address
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.python import components, failure
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.python.test.test_components import RegistryUsingMixin
from twisted.trial.unittest import TestCase
class SSHSessionClientTests(TestCase):
    """
    SSHSessionClient is an obsolete class used to connect standard IO to
    an SSHSession.
    """
    if not cryptography:
        skip = 'cannot run without cryptography'

    def test_dataReceived(self):
        """
        When data is received, it should be sent to the transport.
        """
        client = session.SSHSessionClient()
        client.transport = StubTransport()
        client.dataReceived(b'test data')
        self.assertEqual(client.transport.buf, b'test data')