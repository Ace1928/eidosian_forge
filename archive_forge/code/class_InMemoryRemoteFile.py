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
@implementer(ISFTPFile)
class InMemoryRemoteFile(BytesIO):
    """
    An L{ISFTPFile} which handles all data in memory.
    """

    def __init__(self, name):
        """
        @param name: Name of this file.
        @type name: L{str}
        """
        self.name = name
        BytesIO.__init__(self)

    def writeChunk(self, start, data):
        """
        @see: L{ISFTPFile.writeChunk}
        """
        self.seek(start)
        self.write(data)
        return defer.succeed(self)

    def close(self):
        """
        @see: L{ISFTPFile.writeChunk}

        Keeps data after file was closed to help with testing.
        """
        self._closed = True

    def getAttrs(self):
        pass

    def readChunk(self, offset, length):
        pass

    def setAttrs(self, attrs):
        pass

    def getvalue(self):
        """
        Get current data of file.

        Allow reading data event when file is closed.
        """
        return BytesIO.getvalue(self)