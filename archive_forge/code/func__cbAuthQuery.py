from twisted.words.protocols.jabber import error, sasl, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish, utility, xpath
def _cbAuthQuery(self, iq):
    jid = self.xmlstream.authenticator.jid
    password = self.xmlstream.authenticator.password
    reply = xmlstream.IQ(self.xmlstream, 'set')
    reply.addElement(('jabber:iq:auth', 'query'))
    reply.query.addElement('username', content=jid.user)
    reply.query.addElement('resource', content=jid.resource)
    if DigestAuthQry.matches(iq):
        digest = xmlstream.hashPassword(self.xmlstream.sid, password)
        reply.query.addElement('digest', content=str(digest))
    else:
        reply.query.addElement('password', content=password)
    d = reply.send()
    d.addCallbacks(self._cbAuth, self._ebAuth)
    return d