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
@skipIf(runtime.platform.getType() != 'posix', 'Only runs on POSIX platform')
@skipIf(not interfaces.IReactorProcess(reactor, None), "reactor doesn't support IReactorProcess")
class TwoProcessesPosixTests(TestTwoProcessesBase, unittest.TestCase):

    def tearDown(self):
        for pp, pr in zip(self.pp, self.processes):
            if not pp.finished:
                try:
                    os.kill(pr.pid, signal.SIGTERM)
                except OSError:
                    pass
        return self._onClose()

    def kill(self, num):
        if self.verbose:
            print('kill [%d] with SIGTERM' % num)
        p = self.processes[num]
        pp = self.pp[num]
        self.assertFalse(pp.finished, 'Process finished too early')
        os.kill(p.pid, signal.SIGTERM)
        if self.verbose:
            print(self.pp[0].finished, self.pp[1].finished)

    def test_kill(self):
        if self.verbose:
            print('starting processes')
        self.createProcesses(usePTY=0)
        reactor.callLater(1, self.kill, 0)
        reactor.callLater(2, self.kill, 1)
        return self._onClose()

    def test_closePty(self):
        if self.verbose:
            print('starting processes')
        self.createProcesses(usePTY=1)
        reactor.callLater(1, self.close, 0)
        reactor.callLater(2, self.close, 1)
        return self._onClose()

    def test_killPty(self):
        if self.verbose:
            print('starting processes')
        self.createProcesses(usePTY=1)
        reactor.callLater(1, self.kill, 0)
        reactor.callLater(2, self.kill, 1)
        return self._onClose()