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
class SFTPTestProcess(protocol.ProcessProtocol):
    """
    Protocol for testing cftp. Provides an interface between Python (where all
    the tests are) and the cftp client process (which does the work that is
    being tested).
    """

    def __init__(self, onOutReceived):
        """
        @param onOutReceived: A L{Deferred} to be fired as soon as data is
        received from stdout.
        """
        self.clearBuffer()
        self.onOutReceived = onOutReceived
        self.onProcessEnd = None
        self._expectingCommand = None
        self._processEnded = False

    def clearBuffer(self):
        """
        Clear any buffered data received from stdout. Should be private.
        """
        self.buffer = b''
        self._linesReceived = []
        self._lineBuffer = b''

    def outReceived(self, data):
        """
        Called by Twisted when the cftp client prints data to stdout.
        """
        log.msg('got %r' % data)
        lines = (self._lineBuffer + data).split(b'\n')
        self._lineBuffer = lines.pop(-1)
        self._linesReceived.extend(lines)
        if self.onOutReceived is not None:
            d, self.onOutReceived = (self.onOutReceived, None)
            d.callback(data)
        self.buffer += data
        self._checkForCommand()

    def _checkForCommand(self):
        prompt = b'cftp> '
        if self._expectingCommand and self._lineBuffer == prompt:
            buf = b'\n'.join(self._linesReceived)
            if buf.startswith(prompt):
                buf = buf[len(prompt):]
            self.clearBuffer()
            d, self._expectingCommand = (self._expectingCommand, None)
            d.callback(buf)

    def errReceived(self, data):
        """
        Called by Twisted when the cftp client prints data to stderr.
        """
        log.msg('err: %s' % data)

    def getBuffer(self):
        """
        Return the contents of the buffer of data received from stdout.
        """
        return self.buffer

    def runCommand(self, command):
        """
        Issue the given command via the cftp client. Return a C{Deferred} that
        fires when the server returns a result. Note that the C{Deferred} will
        callback even if the server returns some kind of error.

        @param command: A string containing an sftp command.

        @return: A C{Deferred} that fires when the sftp server returns a
        result. The payload is the server's response string.
        """
        self._expectingCommand = defer.Deferred()
        self.clearBuffer()
        if isinstance(command, str):
            command = command.encode('utf-8')
        self.transport.write(command + b'\n')
        return self._expectingCommand

    def runScript(self, commands):
        """
        Run each command in sequence and return a Deferred that fires when all
        commands are completed.

        @param commands: A list of strings containing sftp commands.

        @return: A C{Deferred} that fires when all commands are completed. The
        payload is a list of response strings from the server, in the same
        order as the commands.
        """
        sem = defer.DeferredSemaphore(1)
        dl = [sem.run(self.runCommand, command) for command in commands]
        return defer.gatherResults(dl)

    def killProcess(self):
        """
        Kill the process if it is still running.

        If the process is still running, sends a KILL signal to the transport
        and returns a C{Deferred} which fires when L{processEnded} is called.

        @return: a C{Deferred}.
        """
        if self._processEnded:
            return defer.succeed(None)
        self.onProcessEnd = defer.Deferred()
        self.transport.signalProcess('KILL')
        return self.onProcessEnd

    def processEnded(self, reason):
        """
        Called by Twisted when the cftp client process ends.
        """
        self._processEnded = True
        if self.onProcessEnd:
            d, self.onProcessEnd = (self.onProcessEnd, None)
            d.callback(None)