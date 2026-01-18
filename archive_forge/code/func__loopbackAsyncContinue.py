import tempfile
from zope.interface import implementer
from twisted.internet import defer, interfaces, main, protocol
from twisted.internet.interfaces import IAddress
from twisted.internet.task import deferLater
from twisted.protocols import policies
from twisted.python import failure
def _loopbackAsyncContinue(ignored, server, serverToClient, client, clientToServer, pumpPolicy):
    clientToServer._notificationDeferred = None
    serverToClient._notificationDeferred = None
    from twisted.internet import reactor
    return deferLater(reactor, 0, _loopbackAsyncBody, server, serverToClient, client, clientToServer, pumpPolicy)