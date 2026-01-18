from twisted.words.protocols.jabber import error, sasl, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish, utility, xpath
class IQAuthInitializer:
    """
    Non-SASL Authentication initializer for the initiating entity.

    This protocol is defined in
    U{JEP-0078<http://www.jabber.org/jeps/jep-0078.html>} and mainly serves for
    compatibility with pre-XMPP-1.0 server implementations.

    @cvar INVALID_USER_EVENT: Token to signal that authentication failed, due
        to invalid username.
    @type INVALID_USER_EVENT: L{str}

    @cvar AUTH_FAILED_EVENT: Token to signal that authentication failed, due to
        invalid password.
    @type AUTH_FAILED_EVENT: L{str}
    """
    INVALID_USER_EVENT = '//event/client/basicauth/invaliduser'
    AUTH_FAILED_EVENT = '//event/client/basicauth/authfailed'

    def __init__(self, xs):
        self.xmlstream = xs

    def initialize(self):
        iq = xmlstream.IQ(self.xmlstream, 'get')
        iq.addElement(('jabber:iq:auth', 'query'))
        jid = self.xmlstream.authenticator.jid
        iq.query.addElement('username', content=jid.user)
        d = iq.send()
        d.addCallbacks(self._cbAuthQuery, self._ebAuthQuery)
        return d

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

    def _ebAuthQuery(self, failure):
        failure.trap(error.StanzaError)
        e = failure.value
        if e.condition == 'not-authorized':
            self.xmlstream.dispatch(e.stanza, self.INVALID_USER_EVENT)
        else:
            self.xmlstream.dispatch(e.stanza, self.AUTH_FAILED_EVENT)
        return failure

    def _cbAuth(self, iq):
        pass

    def _ebAuth(self, failure):
        failure.trap(error.StanzaError)
        self.xmlstream.dispatch(failure.value.stanza, self.AUTH_FAILED_EVENT)
        return failure