from zope.interface import implementer
from twisted.application import service, strports
from twisted.conch import manhole, manhole_ssh, telnet
from twisted.conch.insults import insults
from twisted.conch.ssh import keys
from twisted.cred import checkers, portal
from twisted.internet import protocol
from twisted.python import filepath, usage
class chainedProtocolFactory:

    def __init__(self, namespace):
        self.namespace = namespace

    def __call__(self):
        return insults.ServerProtocol(manhole.ColoredManhole, self.namespace)