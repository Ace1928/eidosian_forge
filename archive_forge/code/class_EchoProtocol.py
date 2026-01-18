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
class EchoProtocol(protocol.ProcessProtocol):
    s = b'1234567' * 1001
    n = 10
    finished = 0
    failure = None

    def __init__(self, onEnded):
        self.onEnded = onEnded
        self.count = 0

    def connectionMade(self):
        assert self.n > 2
        for i in range(self.n - 2):
            self.transport.write(self.s)
        self.transport.writeSequence([self.s, self.s])
        self.buffer = self.s * self.n

    def outReceived(self, data):
        if self.buffer[self.count:self.count + len(data)] != data:
            self.failure = ('wrong bytes received', data, self.count)
            self.transport.closeStdin()
        else:
            self.count += len(data)
            if self.count == len(self.buffer):
                self.transport.closeStdin()

    def processEnded(self, reason):
        self.finished = 1
        if not reason.check(error.ProcessDone):
            self.failure = "process didn't terminate normally: " + str(reason)
        self.onEnded.callback(self)