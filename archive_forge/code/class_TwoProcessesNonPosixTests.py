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
@skipIf(runtime.platform.getType() != 'win32', 'Only runs on Windows')
@skipIf(not interfaces.IReactorProcess(reactor, None), "reactor doesn't support IReactorProcess")
class TwoProcessesNonPosixTests(TestTwoProcessesBase, unittest.TestCase):
    pass