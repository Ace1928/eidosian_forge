import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
class ChildResolveProtocol(protocol.ProcessProtocol):

    def __init__(self, onCompletion):
        self.onCompletion = onCompletion

    def connectionMade(self):
        self.output = []
        self.error = []

    def outReceived(self, out):
        self.output.append(out)

    def errReceived(self, err):
        self.error.append(err)

    def processEnded(self, reason):
        self.onCompletion.callback((reason, self.output, self.error))
        self.onCompletion = None