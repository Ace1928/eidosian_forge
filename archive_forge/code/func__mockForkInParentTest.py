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
def _mockForkInParentTest(self):
    """
        Assert that in the main process, spawnProcess disables the garbage
        collector, calls fork, closes the pipe file descriptors it created for
        the child process, and calls waitpid.
        """
    self.mockos.child = False
    cmd = b'/mock/ouch'
    d = defer.Deferred()
    p = TrivialProcessProtocol(d)
    reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
    self.assertEqual(set(self.mockos.closed), {-1, -4, -6})
    self.assertProcessLaunched()