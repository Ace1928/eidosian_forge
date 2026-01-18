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
class StubProcessProtocol(protocol.ProcessProtocol):
    """
    ProcessProtocol counter-implementation: all methods on this class raise an
    exception, so instances of this may be used to verify that only certain
    methods are called.
    """

    def outReceived(self, data):
        raise NotImplementedError()

    def errReceived(self, data):
        raise NotImplementedError()

    def inConnectionLost(self):
        raise NotImplementedError()

    def outConnectionLost(self):
        raise NotImplementedError()

    def errConnectionLost(self):
        raise NotImplementedError()