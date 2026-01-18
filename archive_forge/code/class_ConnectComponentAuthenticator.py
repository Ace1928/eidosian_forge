from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
class ConnectComponentAuthenticator(xmlstream.ConnectAuthenticator):
    """
    Authenticator to permit an XmlStream to authenticate against a Jabber
    server as an external component (where the Authenticator is initiating the
    stream).
    """
    namespace = NS_COMPONENT_ACCEPT

    def __init__(self, componentjid, password):
        """
        @type componentjid: C{str}
        @param componentjid: Jabber ID that this component wishes to bind to.

        @type password: C{str}
        @param password: Password/secret this component uses to authenticate.
        """
        xmlstream.ConnectAuthenticator.__init__(self, componentjid)
        self.password = password

    def associateWithStream(self, xs):
        xs.version = (0, 0)
        xmlstream.ConnectAuthenticator.associateWithStream(self, xs)
        xs.initializers = [ComponentInitiatingInitializer(xs)]