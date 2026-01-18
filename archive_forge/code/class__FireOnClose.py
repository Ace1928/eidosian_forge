import tempfile
from zope.interface import implementer
from twisted.internet import defer, interfaces, main, protocol
from twisted.internet.interfaces import IAddress
from twisted.internet.task import deferLater
from twisted.protocols import policies
from twisted.python import failure
class _FireOnClose(policies.ProtocolWrapper):

    def __init__(self, protocol, factory):
        policies.ProtocolWrapper.__init__(self, protocol, factory)
        self.deferred = defer.Deferred()

    def connectionLost(self, reason):
        policies.ProtocolWrapper.connectionLost(self, reason)
        self.deferred.callback(None)