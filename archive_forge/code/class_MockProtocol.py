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
class MockProtocol(protocol.Protocol):
    """
    A sample Protocol which stores the data passed to it.

    @ivar packetData: a L{bytes} of data to be sent when the connection is
        made.
    @ivar data: a L{bytes} of the data passed to us.
    @ivar open: True if the channel is open.
    @ivar reason: if not None, the reason the protocol was closed.
    """
    packetData = b''

    def connectionMade(self):
        """
        Set up the instance variables.  If we have any packetData, send it
        along.
        """
        self.data = b''
        self.open = True
        self.reason = None
        if self.packetData:
            self.dataReceived(self.packetData)

    def dataReceived(self, data):
        """
        Store the received data and write it back with a tilde appended.
        The tilde is appended so that the tests can verify that we processed
        the data.
        """
        self.data += data
        self.transport.write(data + b'~')

    def connectionLost(self, reason):
        """
        Close the protocol and store the reason.
        """
        self.open = False
        self.reason = reason