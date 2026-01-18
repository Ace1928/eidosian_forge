import os
import sys
from unittest import skipIf
from twisted.conch import recvline
from twisted.conch.insults import insults
from twisted.cred import portal
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.python import components, filepath, reflect
from twisted.python.compat import iterbytes
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
from twisted.conch import telnet
from twisted.conch.insults import helper
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import checkers
from twisted.conch.test import test_telnet
class SessionChannel(channel.SSHChannel):
    name = b'session'

    def __init__(self, protocolFactory, protocolArgs, protocolKwArgs, width, height, *a, **kw):
        channel.SSHChannel.__init__(self, *a, **kw)
        self.protocolFactory = protocolFactory
        self.protocolArgs = protocolArgs
        self.protocolKwArgs = protocolKwArgs
        self.width = width
        self.height = height

    def channelOpen(self, data):
        term = session.packRequest_pty_req(b'vt102', (self.height, self.width, 0, 0), b'')
        self.conn.sendRequest(self, b'pty-req', term)
        self.conn.sendRequest(self, b'shell', b'')
        self._protocolInstance = self.protocolFactory(*self.protocolArgs, **self.protocolKwArgs)
        self._protocolInstance.factory = self
        self._protocolInstance.makeConnection(self)

    def closed(self):
        self._protocolInstance.connectionLost(error.ConnectionDone())

    def dataReceived(self, data):
        self._protocolInstance.dataReceived(data)