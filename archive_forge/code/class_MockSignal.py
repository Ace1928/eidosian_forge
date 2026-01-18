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
class MockSignal:
    """
    Neuter L{signal.signal}, but pass other attributes unscathed
    """

    def signal(self, sig, action):
        return signal.getsignal(sig)

    def __getattr__(self, attr):
        return getattr(signal, attr)