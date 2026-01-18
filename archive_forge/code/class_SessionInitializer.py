from twisted.words.protocols.jabber import error, sasl, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish, utility, xpath
class SessionInitializer(xmlstream.BaseFeatureInitiatingInitializer):
    """
    Initializer that implements session establishment for the initiating
    entity.

    This protocol is defined in U{RFC 3921, section
    3<http://www.xmpp.org/specs/rfc3921.html#session>}.
    """
    feature = (NS_XMPP_SESSION, 'session')

    def start(self):
        iq = xmlstream.IQ(self.xmlstream, 'set')
        iq.addElement((NS_XMPP_SESSION, 'session'))
        return iq.send()