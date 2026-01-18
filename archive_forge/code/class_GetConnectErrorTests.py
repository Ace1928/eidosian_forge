from __future__ import annotations
import errno
import socket
import sys
from typing import Sequence
from twisted.internet import error
from twisted.trial import unittest
class GetConnectErrorTests(unittest.SynchronousTestCase):
    """
    Given an exception instance thrown by C{socket.connect},
    L{error.getConnectError} returns the appropriate high-level Twisted
    exception instance.
    """

    def assertErrnoException(self, errno: int, expectedClass: type[error.ConnectError]) -> None:
        """
        When called with a tuple with the given errno,
        L{error.getConnectError} returns an exception which is an instance of
        the expected class.
        """
        e = (errno, 'lalala')
        result = error.getConnectError(e)
        self.assertCorrectException(errno, 'lalala', result, expectedClass)

    def assertCorrectException(self, errno: int | None, message: object, result: error.ConnectError, expectedClass: type[error.ConnectError]) -> None:
        """
        The given result of L{error.getConnectError} has the given attributes
        (C{osError} and C{args}), and is an instance of the given class.
        """
        self.assertEqual(result.__class__, expectedClass)
        self.assertEqual(result.osError, errno)
        self.assertEqual(result.args, (message,))

    def test_errno(self) -> None:
        """
        L{error.getConnectError} converts based on errno for C{socket.error}.
        """
        self.assertErrnoException(errno.ENETUNREACH, error.NoRouteError)
        self.assertErrnoException(errno.ECONNREFUSED, error.ConnectionRefusedError)
        self.assertErrnoException(errno.ETIMEDOUT, error.TCPTimedOutError)
        if sys.platform == 'win32':
            self.assertErrnoException(errno.WSAECONNREFUSED, error.ConnectionRefusedError)
            self.assertErrnoException(errno.WSAENETUNREACH, error.NoRouteError)

    def test_gaierror(self) -> None:
        """
        L{error.getConnectError} converts to a L{error.UnknownHostError} given
        a C{socket.gaierror} instance.
        """
        result = error.getConnectError(socket.gaierror(12, 'hello'))
        self.assertCorrectException(12, 'hello', result, error.UnknownHostError)

    def test_nonTuple(self) -> None:
        """
        L{error.getConnectError} converts to a L{error.ConnectError} given
        an argument that cannot be unpacked.
        """
        e = Exception()
        result = error.getConnectError(e)
        self.assertCorrectException(None, e, result, error.ConnectError)