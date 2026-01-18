from zope.interface import directlyProvides
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import ISSLTransport
from twisted.protocols.tls import TLSMemoryBIOFactory
class ClientMixin:
    """
    A mixin for L{twisted.internet.tcp.Client} which just marks it as a client
    for the purposes of the default TLS handshake.

    @ivar _tlsClientDefault: Always C{True}, indicating that this is a client
        connection, and by default when TLS is negotiated this class will act as
        a TLS client.
    """
    _tlsClientDefault = True