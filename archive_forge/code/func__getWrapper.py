import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def _getWrapper(self):
    """
        Return L{policies.ProtocolWrapper} that has been connected to a
        L{StringTransport}.
        """
    wrapper = policies.ProtocolWrapper(policies.WrappingFactory(Server()), protocol.Protocol())
    transport = StringTransport()
    wrapper.makeConnection(transport)
    return wrapper