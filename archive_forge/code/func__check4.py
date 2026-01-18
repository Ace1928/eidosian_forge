import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def _check4(results):
    self.assertEqual(c4.connected, 1)
    self.assertEqual(c4.disconnected, 0)
    return results