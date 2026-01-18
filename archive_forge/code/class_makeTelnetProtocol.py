from zope.interface import implementer
from twisted.application import service, strports
from twisted.conch import manhole, manhole_ssh, telnet
from twisted.conch.insults import insults
from twisted.conch.ssh import keys
from twisted.cred import checkers, portal
from twisted.internet import protocol
from twisted.python import filepath, usage
class makeTelnetProtocol:

    def __init__(self, portal):
        self.portal = portal

    def __call__(self):
        auth = telnet.AuthenticatingTelnetProtocol
        args = (self.portal,)
        return telnet.TelnetTransport(auth, *args)