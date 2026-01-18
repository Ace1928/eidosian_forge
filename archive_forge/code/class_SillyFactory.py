import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
class SillyFactory(protocol.ClientFactory):

    def __init__(self, p):
        self.p = p

    def buildProtocol(self, addr):
        return self.p