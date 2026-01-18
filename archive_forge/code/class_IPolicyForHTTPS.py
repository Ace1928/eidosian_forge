from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
class IPolicyForHTTPS(Interface):
    """
    An L{IPolicyForHTTPS} provides a policy for verifying the certificates of
    HTTPS connections, in the form of a L{client connection creator
    <twisted.internet.interfaces.IOpenSSLClientConnectionCreator>} per network
    location.

    @since: 14.0
    """

    def creatorForNetloc(hostname, port):
        """
        Create a L{client connection creator
        <twisted.internet.interfaces.IOpenSSLClientConnectionCreator>}
        appropriate for the given URL "netloc"; i.e. hostname and port number
        pair.

        @param hostname: The name of the requested remote host.
        @type hostname: L{bytes}

        @param port: The number of the requested remote port.
        @type port: L{int}

        @return: A client connection creator expressing the security
            requirements for the given remote host.
        @rtype: L{client connection creator
            <twisted.internet.interfaces.IOpenSSLClientConnectionCreator>}
        """