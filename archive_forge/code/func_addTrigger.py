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
def addTrigger(self, event, phase, func):
    """
        Add a trigger to the reactor and remember it in C{self.triggers}.
        """
    t = reactor.addSystemEventTrigger(event, phase, func)
    self.triggers.append(t)
    return t