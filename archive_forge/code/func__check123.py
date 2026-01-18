import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def _check123(results):
    self.assertEqual([c.connected for c in (c1, c2, c3)], [1, 1, 1])
    self.assertEqual([c.disconnected for c in (c1, c2, c3)], [0, 0, 1])
    self.assertEqual(len(tServer.protocols.keys()), 2)
    return results