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
def _waitForThread(self):
    """
        The reactor's threadpool is only available when the reactor is running,
        so to have a sane behavior during the tests we make a dummy
        L{threads.deferToThread} call.
        """
    return threads.deferToThread(time.sleep, 0)