from zope.interface import implementer
from twisted.application import service, strports
from twisted.conch import manhole, manhole_ssh, telnet
from twisted.conch.insults import insults
from twisted.conch.ssh import keys
from twisted.cred import checkers, portal
from twisted.internet import protocol
from twisted.python import filepath, usage
@implementer(portal.IRealm)
class _StupidRealm:

    def __init__(self, proto, *a, **kw):
        self.protocolFactory = proto
        self.protocolArgs = a
        self.protocolKwArgs = kw

    def requestAvatar(self, avatarId, *interfaces):
        if telnet.ITelnetProtocol in interfaces:
            return (telnet.ITelnetProtocol, self.protocolFactory(*self.protocolArgs, **self.protocolKwArgs), lambda: None)
        raise NotImplementedError()