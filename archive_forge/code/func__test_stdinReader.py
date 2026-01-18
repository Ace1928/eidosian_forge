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
def _test_stdinReader(self, pyExe, args, env, path):
    """
        Spawn a process, write to stdin, and check the output.
        """
    p = Accumulator()
    d = p.endedDeferred = defer.Deferred()
    reactor.spawnProcess(p, pyExe, args, env, path)
    p.transport.write(b'hello, world')
    p.transport.closeStdin()

    def processEnded(ign):
        self.assertEqual(p.errF.getvalue(), b'err\nerr\n')
        self.assertEqual(p.outF.getvalue(), b'out\nhello, world\nout\n')
    return d.addCallback(processEnded)