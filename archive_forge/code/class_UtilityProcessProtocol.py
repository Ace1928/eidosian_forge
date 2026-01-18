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
class UtilityProcessProtocol(protocol.ProcessProtocol):
    """
    Helper class for launching a Python process and getting a result from it.

    @ivar programName: The name of the program to run.
    """
    programName: bytes = b''

    @classmethod
    def run(cls, reactor, argv, env):
        """
        Run a Python process connected to a new instance of this protocol
        class.  Return the protocol instance.

        The Python process is given C{self.program} on the command line to
        execute, in addition to anything specified by C{argv}.  C{env} is
        the complete environment.
        """
        self = cls()
        reactor.spawnProcess(self, pyExe, [pyExe, '-u', '-m', self.programName] + argv, env=env)
        return self

    def __init__(self):
        self.bytes = []
        self.requests = []

    def parseChunks(self, bytes):
        """
        Called with all bytes received on stdout when the process exits.
        """
        raise NotImplementedError()

    def getResult(self):
        """
        Return a Deferred which will fire with the result of L{parseChunks}
        when the child process exits.
        """
        d = defer.Deferred()
        self.requests.append(d)
        return d

    def _fireResultDeferreds(self, result):
        """
        Callback all Deferreds returned up until now by L{getResult}
        with the given result object.
        """
        requests = self.requests
        self.requests = None
        for d in requests:
            d.callback(result)

    def outReceived(self, bytes):
        """
        Accumulate output from the child process in a list.
        """
        self.bytes.append(bytes)

    def processEnded(self, reason):
        """
        Handle process termination by parsing all received output and firing
        any waiting Deferreds.
        """
        self._fireResultDeferreds(self.parseChunks(self.bytes))