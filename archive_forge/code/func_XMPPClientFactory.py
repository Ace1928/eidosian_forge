from twisted.words.protocols.jabber import error, sasl, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish, utility, xpath
def XMPPClientFactory(jid, password, configurationForTLS=None):
    """
    Client factory for XMPP 1.0 (only).

    This returns a L{xmlstream.XmlStreamFactory} with an L{XMPPAuthenticator}
    object to perform the stream initialization steps (such as authentication).

    @see: The notes at L{XMPPAuthenticator} describe how the C{jid} and
    C{password} parameters are to be used.

    @param jid: Jabber ID to connect with.
    @type jid: L{jid.JID}

    @param password: password to authenticate with.
    @type password: L{unicode}

    @param configurationForTLS: An object which creates appropriately
        configured TLS connections. This is passed to C{startTLS} on the
        transport and is preferably created using
        L{twisted.internet.ssl.optionsForClientTLS}. If L{None}, the default is
        to verify the server certificate against the trust roots as provided by
        the platform. See L{twisted.internet._sslverify.platformTrust}.
    @type configurationForTLS: L{IOpenSSLClientConnectionCreator} or L{None}

    @return: XML stream factory.
    @rtype: L{xmlstream.XmlStreamFactory}
    """
    a = XMPPAuthenticator(jid, password, configurationForTLS=configurationForTLS)
    return xmlstream.XmlStreamFactory(a)