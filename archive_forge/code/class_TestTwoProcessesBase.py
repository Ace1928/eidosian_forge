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
class TestTwoProcessesBase:

    def setUp(self):
        self.processes = [None, None]
        self.pp = [None, None]
        self.done = 0
        self.verbose = 0

    def createProcesses(self, usePTY=0):
        scriptPath = b'twisted.test.process_reader'
        for num in (0, 1):
            self.pp[num] = TwoProcessProtocol()
            self.pp[num].num = num
            p = reactor.spawnProcess(self.pp[num], pyExe, [pyExe, b'-u', b'-m', scriptPath], env=properEnv, usePTY=usePTY)
            self.processes[num] = p

    def close(self, num):
        if self.verbose:
            print('closing stdin [%d]' % num)
        p = self.processes[num]
        pp = self.pp[num]
        self.assertFalse(pp.finished, 'Process finished too early')
        p.loseConnection()
        if self.verbose:
            print(self.pp[0].finished, self.pp[1].finished)

    def _onClose(self):
        return defer.gatherResults([p.deferred for p in self.pp])

    def test_close(self):
        if self.verbose:
            print('starting processes')
        self.createProcesses()
        reactor.callLater(1, self.close, 0)
        reactor.callLater(2, self.close, 1)
        return self._onClose()