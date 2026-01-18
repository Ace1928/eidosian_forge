import itertools
import os
import sys
from unittest import skipIf
from twisted.internet import defer, error, protocol, reactor, stdio
from twisted.python import filepath, log
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SkipTest, TestCase
def _junkPath(self):
    junkPath = self.mktemp()
    with open(junkPath, 'wb') as junkFile:
        for i in range(1024):
            junkFile.write(b'%d\n' % (i,))
    return junkPath