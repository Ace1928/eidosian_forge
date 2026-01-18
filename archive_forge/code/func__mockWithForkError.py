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
def _mockWithForkError(self):
    """
        Assert that if the fork call fails, no other process setup calls are
        made and that spawnProcess raises the exception fork raised.
        """
    self.mockos.raiseFork = OSError(errno.EAGAIN, None)
    protocol = TrivialProcessProtocol(None)
    self.assertRaises(OSError, reactor.spawnProcess, protocol, None)
    self.assertEqual(self.mockos.actions, [('fork', False)])