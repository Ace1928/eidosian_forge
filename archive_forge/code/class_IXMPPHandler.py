from zope.interface import Attribute, Interface
class IXMPPHandler(Interface):
    """
    Interface for XMPP protocol handlers.

    Objects that provide this interface can be added to a stream manager to
    handle of (part of) an XMPP extension protocol.
    """
    parent = Attribute('XML stream manager for this handler')
    xmlstream = Attribute('The managed XML stream')

    def setHandlerParent(parent):
        """
        Set the parent of the handler.

        @type parent: L{IXMPPHandlerCollection}
        """

    def disownHandlerParent(parent):
        """
        Remove the parent of the handler.

        @type parent: L{IXMPPHandlerCollection}
        """

    def makeConnection(xs):
        """
        A connection over the underlying transport of the XML stream has been
        established.

        At this point, no traffic has been exchanged over the XML stream
        given in C{xs}.

        This should setup L{xmlstream} and call L{connectionMade}.

        @type xs:
               L{twisted.words.protocols.jabber.xmlstream.XmlStream}
        """

    def connectionMade():
        """
        Called after a connection has been established.

        This method can be used to change properties of the XML Stream, its
        authenticator or the stream manager prior to stream initialization
        (including authentication).
        """

    def connectionInitialized():
        """
        The XML stream has been initialized.

        At this point, authentication was successful, and XML stanzas can be
        exchanged over the XML stream L{xmlstream}. This method can be
        used to setup observers for incoming stanzas.
        """

    def connectionLost(reason):
        """
        The XML stream has been closed.

        Subsequent use of C{parent.send} will result in data being queued
        until a new connection has been established.

        @type reason: L{twisted.python.failure.Failure}
        """