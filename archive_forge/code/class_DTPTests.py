import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class DTPTests(TestCase):
    """
    Tests for L{ftp.DTP}.

    The DTP instances in these tests are generated using
    DTPFactory.buildProtocol()
    """

    def setUp(self):
        """
        Create a fake protocol interpreter, a L{ftp.DTPFactory} instance,
        and dummy transport to help with tests.
        """
        self.reactor = task.Clock()

        class ProtocolInterpreter:
            dtpInstance = None
        self.protocolInterpreter = ProtocolInterpreter()
        self.factory = ftp.DTPFactory(self.protocolInterpreter, None, self.reactor)
        self.transport = proto_helpers.StringTransportWithDisconnection()

    def test_sendLineNewline(self):
        """
        L{ftp.DTP.sendLine} writes the line passed to it plus a line delimiter
        to its transport.
        """
        dtpInstance = self.factory.buildProtocol(None)
        dtpInstance.makeConnection(self.transport)
        lineContent = b'line content'
        dtpInstance.sendLine(lineContent)
        dataSent = self.transport.value()
        self.assertEqual(lineContent + b'\r\n', dataSent)