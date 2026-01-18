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
class ClosingPipesProcessProtocol(protocol.ProcessProtocol):
    output = b''
    errput = b''

    def __init__(self, outOrErr):
        self.deferred = defer.Deferred()
        self.outOrErr = outOrErr

    def processEnded(self, reason):
        self.deferred.callback(reason)

    def outReceived(self, data):
        self.output += data

    def errReceived(self, data):
        self.errput += data