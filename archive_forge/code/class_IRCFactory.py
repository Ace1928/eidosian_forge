from time import ctime, time
from zope.interface import implementer
from twisted import copyright
from twisted.cred import credentials, error as ecred, portal
from twisted.internet import defer, protocol
from twisted.python import failure, log, reflect
from twisted.python.components import registerAdapter
from twisted.spread import pb
from twisted.words import ewords, iwords
from twisted.words.protocols import irc
class IRCFactory(protocol.ServerFactory):
    """
    IRC server that creates instances of the L{IRCUser} protocol.

    @ivar _serverInfo: A dictionary mapping:
        "serviceName" to the name of the server,
        "serviceVersion" to the copyright version,
        "creationDate" to the time that the server was started.
    """
    protocol = IRCUser

    def __init__(self, realm, portal):
        self.realm = realm
        self.portal = portal
        self._serverInfo = {'serviceName': self.realm.name, 'serviceVersion': copyright.version, 'creationDate': ctime()}