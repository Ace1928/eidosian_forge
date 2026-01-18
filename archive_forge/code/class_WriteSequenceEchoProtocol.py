import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
class WriteSequenceEchoProtocol(EchoProtocol):

    def dataReceived(self, bytes):
        if bytes.find(b'vector!') != -1:
            self.transport.writeSequence([bytes])
        else:
            EchoProtocol.dataReceived(self, bytes)