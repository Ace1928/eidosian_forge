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
def _endProcess(self, reason, p):
    """
        Check that a failed write prevented the process from getting to its
        custom exit code.
        """
    self.assertNotEqual(reason.exitCode, 42, 'process reason was %r' % reason)
    self.assertEqual(p.output, b'')
    return p.errput