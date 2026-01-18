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
class _PassiveConnectionFactory(protocol.ClientFactory):
    noisy = False

    def __init__(self, protoInstance):
        self.protoInstance = protoInstance

    def buildProtocol(self, ignored):
        self.protoInstance.factory = self
        return self.protoInstance

    def clientConnectionFailed(self, connector, reason):
        e = FTPError('Connection Failed', reason)
        self.protoInstance.deferred.errback(e)