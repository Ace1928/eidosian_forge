import errno
import gc
import io
import os
import socket
from functools import wraps
from typing import Callable, ClassVar, List, Mapping, Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass, verifyObject
import attr
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import (
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import (
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.test.test_tcp import (
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
@skipIf(SKIP_EMFILE, 'Reserved EMFILE file descriptor not supported on Windows.')
class ExhaustsFileDescriptorsTests(SynchronousTestCase):
    """
    Tests for L{_ExhaustsFileDescriptors}.
    """

    def setUp(self):
        self.exhauster = _ExhaustsFileDescriptors()
        self.addCleanup(self.exhauster.release)

    def openAFile(self):
        """
        Attempt to open a file; if successful, the file is immediately
        closed.
        """
        open(os.devnull).close()

    def test_providesInterface(self):
        """
        L{_ExhaustsFileDescriptors} instances provide
        L{_IExhaustsFileDescriptors}.
        """
        verifyObject(_IExhaustsFileDescriptors, self.exhauster)

    def test_count(self):
        """
        L{_ExhaustsFileDescriptors.count} returns the number of open
        file descriptors.
        """
        self.assertEqual(self.exhauster.count(), 0)
        self.exhauster.exhaust()
        self.assertGreater(self.exhauster.count(), 0)
        self.exhauster.release()
        self.assertEqual(self.exhauster.count(), 0)

    def test_exhaustTriggersEMFILE(self):
        """
        L{_ExhaustsFileDescriptors.exhaust} causes the process to
        exhaust its available file descriptors.
        """
        self.addCleanup(self.exhauster.release)
        self.exhauster.exhaust()
        exception = self.assertRaises(IOError, self.openAFile)
        self.assertEqual(exception.errno, errno.EMFILE)

    def test_exhaustRaisesOSError(self):
        """
        An L{OSError} raised within
        L{_ExhaustsFileDescriptors.exhaust} with an C{errno} other
        than C{EMFILE} is reraised to the caller.
        """

        def raiseOSError():
            raise OSError(errno.EMFILE + 1, 'Not EMFILE')
        exhauster = _ExhaustsFileDescriptors(raiseOSError)
        self.assertRaises(OSError, exhauster.exhaust)

    def test_release(self):
        """
        L{_ExhaustsFileDescriptors.release} releases all opened
        file descriptors.
        """
        self.exhauster.exhaust()
        self.exhauster.release()
        self.openAFile()

    def test_fileDescriptorsReleasedOnFailure(self):
        """
        L{_ExhaustsFileDescriptors.exhaust} closes any opened file
        descriptors if an exception occurs during its exhaustion loop.
        """
        fileDescriptors = []

        def failsAfterThree():
            if len(fileDescriptors) == 3:
                raise ValueError('test_fileDescriptorsReleasedOnFailure fake open exception')
            else:
                fd = os.dup(0)
                fileDescriptors.append(fd)
                return fd
        exhauster = _ExhaustsFileDescriptors(failsAfterThree)
        self.addCleanup(exhauster.release)
        self.assertRaises(ValueError, exhauster.exhaust)
        self.assertEqual(len(fileDescriptors), 3)
        self.assertEqual(exhauster.count(), 0)
        for fd in fileDescriptors:
            exception = self.assertRaises(OSError, os.fstat, fd)
            self.assertEqual(exception.errno, errno.EBADF)

    def test_releaseIgnoresEBADF(self):
        """
        L{_ExhaustsFileDescriptors.release} continues to close opened
        file descriptors even when closing one fails with C{EBADF}.
        """
        fileDescriptors = []

        def recordFileDescriptors():
            fd = os.dup(0)
            fileDescriptors.append(fd)
            return fd
        exhauster = _ExhaustsFileDescriptors(recordFileDescriptors)
        self.addCleanup(exhauster.release)
        exhauster.exhaust()
        self.assertGreater(exhauster.count(), 0)
        os.close(fileDescriptors[0])
        exhauster.release()
        self.assertEqual(exhauster.count(), 0)

    def test_releaseRaisesOSError(self):
        """
        An L{OSError} raised within
        L{_ExhaustsFileDescriptors.release} with an C{errno} other than
        C{EBADF} is reraised to the caller.
        """
        fakeFileDescriptors = []

        def opensThree():
            if len(fakeFileDescriptors) == 3:
                raise OSError(errno.EMFILE, 'Too many files')
            fakeFileDescriptors.append(-1)
            return fakeFileDescriptors[-1]

        def failingClose(fd):
            raise OSError(11, 'test_releaseRaisesOSError fake OSError')
        exhauster = _ExhaustsFileDescriptors(opensThree, close=failingClose)
        self.assertEqual(exhauster.count(), 0)
        exhauster.exhaust()
        self.assertGreater(exhauster.count(), 0)
        self.assertRaises(OSError, exhauster.release)