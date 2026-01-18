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
def addTimer(self, when, callback):
    self.timers[self.counter] = reactor.callLater(when * 0.01, callback, self.counter)
    self.counter += 1
    self.checkTimers()