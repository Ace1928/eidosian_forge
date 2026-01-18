import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
class Accumulator(protocol.ProcessProtocol):
    """Accumulate data from a process."""
    closed = 0
    endedDeferred = None

    def connectionMade(self):
        self.outF = BytesIO()
        self.errF = BytesIO()

    def outReceived(self, d):
        self.outF.write(d)

    def errReceived(self, d):
        self.errF.write(d)

    def outConnectionLost(self):
        pass

    def errConnectionLost(self):
        pass

    def processEnded(self, reason):
        self.closed = 1
        if self.endedDeferred is not None:
            d, self.endedDeferred = (self.endedDeferred, None)
            d.callback(None)