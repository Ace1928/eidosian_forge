from __future__ import annotations
import errno
import socket
import sys
from typing import Sequence
from twisted.internet import error
from twisted.trial import unittest
class StringificationTests(unittest.SynchronousTestCase):
    """Test that the exceptions have useful stringifications."""
    listOfTests: list[tuple[str, type[Exception], Sequence[str | int | Exception | None], dict[str, str | int]]] = [('An error occurred binding to an interface.', error.BindError, [], {}), ('An error occurred binding to an interface: foo.', error.BindError, ['foo'], {}), ('An error occurred binding to an interface: foo bar.', error.BindError, ['foo', 'bar'], {}), ("Couldn't listen on eth0:4242: Foo.", error.CannotListenError, ('eth0', 4242, socket.error('Foo')), {}), ('Message is too long to send.', error.MessageLengthError, [], {}), ('Message is too long to send: foo bar.', error.MessageLengthError, ['foo', 'bar'], {}), ('DNS lookup failed.', error.DNSLookupError, [], {}), ('DNS lookup failed: foo bar.', error.DNSLookupError, ['foo', 'bar'], {}), ('An error occurred while connecting.', error.ConnectError, [], {}), ('An error occurred while connecting: someOsError.', error.ConnectError, ['someOsError'], {}), ('An error occurred while connecting: foo.', error.ConnectError, [], {'string': 'foo'}), ('An error occurred while connecting: someOsError: foo.', error.ConnectError, ['someOsError', 'foo'], {}), ("Couldn't bind.", error.ConnectBindError, [], {}), ("Couldn't bind: someOsError.", error.ConnectBindError, ['someOsError'], {}), ("Couldn't bind: someOsError: foo.", error.ConnectBindError, ['someOsError', 'foo'], {}), ("Hostname couldn't be looked up.", error.UnknownHostError, [], {}), ('No route to host.', error.NoRouteError, [], {}), ('Connection was refused by other side.', error.ConnectionRefusedError, [], {}), ('TCP connection timed out.', error.TCPTimedOutError, [], {}), ('File used for UNIX socket is no good.', error.BadFileError, [], {}), ('Service name given as port is unknown.', error.ServiceNameUnknownError, [], {}), ('User aborted connection.', error.UserError, [], {}), ('User timeout caused connection failure.', error.TimeoutError, [], {}), ('An SSL error occurred.', error.SSLError, [], {}), ('Connection to the other side was lost in a non-clean fashion.', error.ConnectionLost, [], {}), ('Connection to the other side was lost in a non-clean fashion: foo bar.', error.ConnectionLost, ['foo', 'bar'], {}), ('Connection was closed cleanly.', error.ConnectionDone, [], {}), ('Connection was closed cleanly: foo bar.', error.ConnectionDone, ['foo', 'bar'], {}), ('Uh.', error.ConnectionFdescWentAway, [], {}), ('Tried to cancel an already-called event.', error.AlreadyCalled, [], {}), ('Tried to cancel an already-called event: foo bar.', error.AlreadyCalled, ['foo', 'bar'], {}), ('Tried to cancel an already-cancelled event.', error.AlreadyCancelled, [], {}), ('Tried to cancel an already-cancelled event: x 2.', error.AlreadyCancelled, ['x', '2'], {}), ('A process has ended without apparent errors: process finished with exit code 0.', error.ProcessDone, [None], {}), ('A process has ended with a probable error condition: process ended.', error.ProcessTerminated, [], {}), ('A process has ended with a probable error condition: process ended with exit code 42.', error.ProcessTerminated, [], {'exitCode': 42}), ('A process has ended with a probable error condition: process ended by signal SIGBUS.', error.ProcessTerminated, [], {'signal': 'SIGBUS'}), ('The Connector was not connecting when it was asked to stop connecting.', error.NotConnectingError, [], {}), ('The Connector was not connecting when it was asked to stop connecting: x 13.', error.NotConnectingError, ['x', '13'], {}), ('The Port was not listening when it was asked to stop listening.', error.NotListeningError, [], {}), ('The Port was not listening when it was asked to stop listening: a 12.', error.NotListeningError, ['a', '12'], {})]

    def testThemAll(self) -> None:
        for entry in self.listOfTests:
            output = entry[0]
            exception = entry[1]
            args = entry[2]
            kwargs = entry[3]
            self.assertEqual(str(exception(*args, **kwargs)), output)

    def test_connectingCancelledError(self) -> None:
        """
        L{error.ConnectingCancelledError} has an C{address} attribute.
        """
        address = object()
        e = error.ConnectingCancelledError(address)
        self.assertIs(e.address, address)