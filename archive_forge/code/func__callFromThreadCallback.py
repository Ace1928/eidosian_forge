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
def _callFromThreadCallback(self, d):
    reactor.callFromThread(self._callFromThreadCallback2, d)
    reactor.callLater(0, self._stopCallFromThreadCallback)