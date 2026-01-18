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
class FilesystemAccessExpectations:
    """
    A test helper used to support expected filesystem access.
    """

    def __init__(self):
        self._cache = {}

    def put(self, path, flags, stream):
        """

        @param path: Path at which the stream is requested.
        @type path: L{str}

        @param path: Flags with which the stream is requested.
        @type path: L{str}

        @param stream: A stream.
        @type stream: C{File}
        """
        self._cache[path, flags] = stream

    def pop(self, path, flags):
        """
        Remove a stream from the memory.

        @param path: Path at which the stream is requested.
        @type path: L{str}

        @param path: Flags with which the stream is requested.
        @type path: L{str}

        @return: A stream.
        @rtype: C{File}
        """
        return self._cache.pop((path, flags))