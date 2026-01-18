import os
import sys
import time
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, threads
from twisted.python import failure, log, threadable, threadpool
from twisted.trial.unittest import TestCase
import time
import %(reactor)s
from twisted.internet import reactor
class ThreadStartupProcessProtocol(protocol.ProcessProtocol):

    def __init__(self, finished):
        self.finished = finished
        self.out = []
        self.err = []

    def outReceived(self, out):
        self.out.append(out)

    def errReceived(self, err):
        self.err.append(err)

    def processEnded(self, reason):
        self.finished.callback((self.out, self.err, reason))