import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
@implementer(IFinishableConsumer)
class SenderProtocol(protocol.Protocol):

    def __init__(self):
        self.connectedDeferred = defer.Deferred()
        self.deferred = defer.Deferred()

    def dataReceived(self, data):
        raise UnexpectedData('Received data from the server on a send-only data-connection')

    def makeConnection(self, transport):
        protocol.Protocol.makeConnection(self, transport)
        self.connectedDeferred.callback(self)

    def connectionLost(self, reason):
        if reason.check(error.ConnectionDone):
            self.deferred.callback('connection done')
        else:
            self.deferred.errback(reason)

    def write(self, data):
        self.transport.write(data)

    def registerProducer(self, producer, streaming):
        """
        Register the given producer with our transport.
        """
        self.transport.registerProducer(producer, streaming)

    def unregisterProducer(self):
        """
        Unregister the previously registered producer.
        """
        self.transport.unregisterProducer()

    def finish(self):
        self.transport.loseConnection()