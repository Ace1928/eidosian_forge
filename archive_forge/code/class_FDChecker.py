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
class FDChecker(protocol.ProcessProtocol):
    state = 0
    data = b''
    failed = None

    def __init__(self, d):
        self.deferred = d

    def fail(self, why):
        self.failed = why
        self.deferred.callback(None)

    def connectionMade(self):
        self.transport.writeToChild(0, b'abcd')
        self.state = 1

    def childDataReceived(self, childFD, data):
        if self.state == 1:
            if childFD != 1:
                self.fail("read '%s' on fd %d (not 1) during state 1" % (childFD, data))
                return
            self.data += data
            if len(self.data) == 6:
                if self.data != b'righto':
                    self.fail("got '%s' on fd1, expected 'righto'" % self.data)
                    return
                self.data = b''
                self.state = 2
                self.transport.writeToChild(3, b'efgh')
                return
        if self.state == 2:
            self.fail(f"read '{childFD}' on fd {data} during state 2")
            return
        if self.state == 3:
            if childFD != 1:
                self.fail(f"read '{childFD}' on fd {data} (not 1) during state 3")
                return
            self.data += data
            if len(self.data) == 6:
                if self.data != b'closed':
                    self.fail("got '%s' on fd1, expected 'closed'" % self.data)
                    return
                self.state = 4
            return
        if self.state == 4:
            self.fail(f"read '{childFD}' on fd {data} during state 4")
            return

    def childConnectionLost(self, childFD):
        if self.state == 1:
            self.fail('got connectionLost(%d) during state 1' % childFD)
            return
        if self.state == 2:
            if childFD != 4:
                self.fail('got connectionLost(%d) (not 4) during state 2' % childFD)
                return
            self.state = 3
            self.transport.closeChildFD(5)
            return

    def processEnded(self, status):
        rc = status.value.exitCode
        if self.state != 4:
            self.fail('processEnded early, rc %d' % rc)
            return
        if status.value.signal != None:
            self.fail('processEnded with signal %s' % status.value.signal)
            return
        if rc != 0:
            self.fail('processEnded with rc %d' % rc)
            return
        self.deferred.callback(None)