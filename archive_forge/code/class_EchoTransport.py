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
class EchoTransport:
    """
    A transport for a ProcessProtocol which echos data that is sent to it with
    a Window newline (CR LF) appended to it.  If a null byte is in the data,
    disconnect.  When we are asked to disconnect, disconnect the
    C{ProcessProtocol} with a 0 exit code.

    @ivar proto: the C{ProcessProtocol} connected to us.
    @ivar data: a L{bytes} of data written to us.
    """

    def __init__(self, processProtocol):
        """
        Initialize our instance variables.

        @param processProtocol: a C{ProcessProtocol} to connect to ourself.
        """
        self.proto = processProtocol
        self.closed = False
        self.data = b''
        processProtocol.makeConnection(self)

    def write(self, data):
        """
        We got some data.  Give it back to our C{ProcessProtocol} with
        a newline attached.  Disconnect if there's a null byte.
        """
        self.data += data
        self.proto.outReceived(data)
        self.proto.outReceived(b'\r\n')
        if b'\x00' in data:
            self.loseConnection()

    def loseConnection(self):
        """
        If we're asked to disconnect (and we haven't already) shut down
        the C{ProcessProtocol} with a 0 exit code.
        """
        if self.closed:
            return
        self.closed = 1
        self.proto.inConnectionLost()
        self.proto.outConnectionLost()
        self.proto.errConnectionLost()
        self.proto.processEnded(failure.Failure(error.ProcessTerminated(0, None, None)))