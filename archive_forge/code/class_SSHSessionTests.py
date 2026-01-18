import getpass
import locale
import operator
import os
import struct
import sys
import time
from io import BytesIO, TextIOWrapper
from unittest import skipIf
from zope.interface import implementer
from twisted.conch import ls
from twisted.conch.interfaces import ISFTPFile
from twisted.conch.test.test_filetransfer import FileTransferTestAvatar, SFTPTestBase
from twisted.cred import portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.internet.utils import getProcessOutputAndValue, getProcessValue
from twisted.python import log
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
@skipIf(skipTests, "don't run w/o spawnProcess or cryptography")
class SSHSessionTests(TestCase):
    """
    Tests for L{twisted.conch.scripts.cftp.SSHSession}.
    """

    def setUp(self) -> None:
        self.stdio = FakeStdio()
        self.channel = SSHSession()
        self.channel.stdio = self.stdio
        self.stderrBuffer = BytesIO()
        self.stderr = TextIOWrapper(self.stderrBuffer)
        self.channel.stderr = self.stderr

    def test_eofReceived(self) -> None:
        """
        L{twisted.conch.scripts.cftp.SSHSession.eofReceived} loses the write
        half of its stdio connection.
        """
        self.channel.eofReceived()
        self.assertTrue(self.stdio.writeConnLost)

    def test_extReceivedStderr(self) -> None:
        """
        L{twisted.conch.scripts.cftp.SSHSession.extReceived} decodes
        stderr data using UTF-8 with the "backslashescape" error handling and
        writes the result to its own stderr.
        """
        errorText = '☃'
        errorBytes = errorText.encode('utf-8')
        self.channel.extReceived(EXTENDED_DATA_STDERR, errorBytes + b'\xff')
        self.assertEqual(self.stderrBuffer.getvalue(), errorBytes + b'\\xff')