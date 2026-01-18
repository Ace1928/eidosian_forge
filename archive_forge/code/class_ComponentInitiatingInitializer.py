from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
class ComponentInitiatingInitializer:
    """
    External server-side component authentication initializer for the
    initiating entity.

    @ivar xmlstream: XML stream between server and component.
    @type xmlstream: L{xmlstream.XmlStream}
    """

    def __init__(self, xs):
        self.xmlstream = xs
        self._deferred = None

    def initialize(self):
        xs = self.xmlstream
        hs = domish.Element((self.xmlstream.namespace, 'handshake'))
        digest = xmlstream.hashPassword(xs.sid, xs.authenticator.password)
        hs.addContent(str(digest))
        xs.addOnetimeObserver('/handshake', self._cbHandshake)
        xs.send(hs)
        self._deferred = defer.Deferred()
        return self._deferred

    def _cbHandshake(self, _):
        self.xmlstream.thisEntity = self.xmlstream.otherEntity
        self._deferred.callback(None)