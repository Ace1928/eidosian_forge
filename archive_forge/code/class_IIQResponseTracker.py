from zope.interface import Attribute, Interface
class IIQResponseTracker(Interface):
    """
    IQ response tracker interface.

    The XMPP stanza C{iq} has a request-response nature that fits
    naturally with deferreds. You send out a request and when the response
    comes back a deferred is fired.

    The L{twisted.words.protocols.jabber.client.IQ} class implements a C{send}
    method that returns a deferred. This deferred is put in a dictionary that
    is kept in an L{XmlStream} object, keyed by the request stanzas C{id}
    attribute.

    An object providing this interface (usually an instance of L{XmlStream}),
    keeps the said dictionary and sets observers on the iq stanzas of type
    C{result} and C{error} and lets the callback fire the associated deferred.
    """
    iqDeferreds = Attribute('Dictionary of deferreds waiting for an iq response')