from twisted.words.protocols.jabber import error, sasl, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish, utility, xpath
class BindInitializer(xmlstream.BaseFeatureInitiatingInitializer):
    """
    Initializer that implements Resource Binding for the initiating entity.

    This protocol is documented in U{RFC 3920, section
    7<http://www.xmpp.org/specs/rfc3920.html#bind>}.
    """
    feature = (NS_XMPP_BIND, 'bind')

    def start(self):
        iq = xmlstream.IQ(self.xmlstream, 'set')
        bind = iq.addElement((NS_XMPP_BIND, 'bind'))
        resource = self.xmlstream.authenticator.jid.resource
        if resource:
            bind.addElement('resource', content=resource)
        d = iq.send()
        d.addCallback(self.onBind)
        return d

    def onBind(self, iq):
        if iq.bind:
            self.xmlstream.authenticator.jid = JID(str(iq.bind.jid))