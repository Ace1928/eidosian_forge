from zope.interface import directlyProvides
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import ISSLTransport
from twisted.protocols.tls import TLSMemoryBIOFactory

        Unregister a producer.

        If TLS is enabled, the TLS connection handles this.
        