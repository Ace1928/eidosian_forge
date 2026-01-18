from zope.interface import directlyProvides
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import ISSLTransport
from twisted.protocols.tls import TLSMemoryBIOFactory
class _BypassTLS:
    """
    L{_BypassTLS} is used as the transport object for the TLS protocol object
    used to implement C{startTLS}.  Its methods skip any TLS logic which
    C{startTLS} enables.

    @ivar _base: A transport class L{_BypassTLS} has been mixed in with to which
        methods will be forwarded.  This class is only responsible for sending
        bytes over the connection, not doing TLS.

    @ivar _connection: A L{Connection} which TLS has been started on which will
        be proxied to by this object.  Any method which has its behavior
        altered after C{startTLS} will be skipped in favor of the base class's
        implementation.  This allows the TLS protocol object to have direct
        access to the transport, necessary to actually implement TLS.
    """

    def __init__(self, base, connection):
        self._base = base
        self._connection = connection

    def __getattr__(self, name):
        """
        Forward any extra attribute access to the original transport object.
        For example, this exposes C{getHost}, the behavior of which does not
        change after TLS is enabled.
        """
        return getattr(self._connection, name)

    def write(self, data):
        """
        Write some bytes directly to the connection.
        """
        return self._base.write(self._connection, data)

    def writeSequence(self, iovec):
        """
        Write a some bytes directly to the connection.
        """
        return self._base.writeSequence(self._connection, iovec)

    def loseConnection(self, *args, **kwargs):
        """
        Close the underlying connection.
        """
        return self._base.loseConnection(self._connection, *args, **kwargs)

    def registerProducer(self, producer, streaming):
        """
        Register a producer with the underlying connection.
        """
        return self._base.registerProducer(self._connection, producer, streaming)

    def unregisterProducer(self):
        """
        Unregister a producer with the underlying connection.
        """
        return self._base.unregisterProducer(self._connection)