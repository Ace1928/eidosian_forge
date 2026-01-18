import os
import signal
import struct
import sys
from unittest import skipIf
from zope.interface import implementer
from twisted.internet import defer, error, protocol
from twisted.internet.address import IPv4Address
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.python import components, failure
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.python.test.test_components import RegistryUsingMixin
from twisted.trial.unittest import TestCase
class StubConnection:
    """
    A stub for twisted.conch.ssh.connection.SSHConnection.  Record the data
    that channels send, and when they try to close the connection.

    @ivar data: a L{dict} mapping C{SSHChannel}s to a C{list} of L{bytes} of
        data they sent.
    @ivar extData: a L{dict} mapping L{SSHChannel}s to a C{list} of L{tuple} of
        (L{int}, L{bytes}) of extended data they sent.
    @ivar requests: a L{dict} mapping L{SSHChannel}s to a C{list} of L{tuple}
        of (L{str}, L{bytes}) of channel requests they made.
    @ivar eofs: a L{dict} mapping L{SSHChannel}s to C{true} if they have sent
        an EOF.
    @ivar closes: a L{dict} mapping L{SSHChannel}s to C{true} if they have sent
        a close.
    """

    def __init__(self, transport=None):
        """
        Initialize our instance variables.
        """
        self.data = {}
        self.extData = {}
        self.requests = {}
        self.eofs = {}
        self.closes = {}
        self.transport = transport

    def logPrefix(self):
        """
        Return our logging prefix.
        """
        return 'MockConnection'

    def sendData(self, channel, data):
        """
        Record the sent data.
        """
        if self.closes.get(channel):
            return
        self.data.setdefault(channel, []).append(data)

    def sendExtendedData(self, channel, type, data):
        """
        Record the sent extended data.
        """
        if self.closes.get(channel):
            return
        self.extData.setdefault(channel, []).append((type, data))

    def sendRequest(self, channel, request, data, wantReply=False):
        """
        Record the sent channel request.
        """
        if self.closes.get(channel):
            return
        self.requests.setdefault(channel, []).append((request, data, wantReply))
        if wantReply:
            return defer.succeed(None)

    def sendEOF(self, channel):
        """
        Record the sent EOF.
        """
        if self.closes.get(channel):
            return
        self.eofs[channel] = True

    def sendClose(self, channel):
        """
        Record the sent close.
        """
        self.closes[channel] = True