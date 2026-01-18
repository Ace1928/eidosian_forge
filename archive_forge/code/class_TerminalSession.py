from typing import Dict
from zope.interface import implementer
from twisted.conch import avatar, error as econch, interfaces as iconch
from twisted.conch.insults import insults
from twisted.conch.ssh import factory, session
from twisted.python import components
@implementer(iconch.ISession)
class TerminalSession(components.Adapter):
    transportFactory = TerminalSessionTransport
    chainedProtocolFactory = insults.ServerProtocol

    def getPty(self, term, windowSize, attrs):
        self.height, self.width = windowSize[:2]

    def openShell(self, proto):
        self.transportFactory(proto, self.chainedProtocolFactory(), iconch.IConchUser(self.original), self.width, self.height)

    def execCommand(self, proto, cmd):
        raise econch.ConchError('Cannot execute commands')

    def windowChanged(self, newWindowSize):
        raise NotImplementedError('Unimplemented: TerminalSession.windowChanged')

    def eofReceived(self):
        raise NotImplementedError('Unimplemented: TerminalSession.eofReceived')

    def closed(self):
        pass