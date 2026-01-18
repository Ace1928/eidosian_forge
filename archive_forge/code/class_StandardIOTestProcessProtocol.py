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
class StandardIOTestProcessProtocol(protocol.ProcessProtocol):
    """
    Test helper for collecting output from a child process and notifying
    something when it exits.

    @ivar onConnection: A L{defer.Deferred} which will be called back with
    L{None} when the connection to the child process is established.

    @ivar onCompletion: A L{defer.Deferred} which will be errbacked with the
    failure associated with the child process exiting when it exits.

    @ivar onDataReceived: A L{defer.Deferred} which will be called back with
    this instance whenever C{childDataReceived} is called, or L{None} to
    suppress these callbacks.

    @ivar data: A C{dict} mapping file descriptors to strings containing all
    bytes received from the child process on each file descriptor.
    """
    onDataReceived = None

    def __init__(self):
        self.onConnection = defer.Deferred()
        self.onCompletion = defer.Deferred()
        self.data = {}

    def connectionMade(self):
        self.onConnection.callback(None)

    def childDataReceived(self, name, bytes):
        """
        Record all bytes received from the child process in the C{data}
        dictionary.  Fire C{onDataReceived} if it is not L{None}.
        """
        self.data[name] = self.data.get(name, b'') + bytes
        if self.onDataReceived is not None:
            d, self.onDataReceived = (self.onDataReceived, None)
            d.callback(self)

    def processEnded(self, reason):
        self.onCompletion.callback(reason)