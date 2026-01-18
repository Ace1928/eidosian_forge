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
class HelpersTests(TestCase):
    """
    Tests for the 4 helper functions: parseRequest_* and packRequest_*.
    """
    if not cryptography:
        skip = 'cannot run without cryptography'

    def test_parseRequest_pty_req(self):
        """
        The payload of a pty-req message is::
            string  terminal
            uint32  columns
            uint32  rows
            uint32  x pixels
            uint32  y pixels
            string  modes

        Modes are::
            byte    mode number
            uint32  mode value
        """
        self.assertEqual(session.parseRequest_pty_req(common.NS(b'xterm') + struct.pack('>4L', 1, 2, 3, 4) + common.NS(struct.pack('>BL', 5, 6))), (b'xterm', (2, 1, 3, 4), [(5, 6)]))

    def test_packRequest_pty_req_old(self):
        """
        See test_parseRequest_pty_req for the payload format.
        """
        packed = session.packRequest_pty_req(b'xterm', (2, 1, 3, 4), b'\x05\x00\x00\x00\x06')
        self.assertEqual(packed, common.NS(b'xterm') + struct.pack('>4L', 1, 2, 3, 4) + common.NS(struct.pack('>BL', 5, 6)))

    def test_packRequest_pty_req(self):
        """
        See test_parseRequest_pty_req for the payload format.
        """
        packed = session.packRequest_pty_req(b'xterm', (2, 1, 3, 4), b'\x05\x00\x00\x00\x06')
        self.assertEqual(packed, common.NS(b'xterm') + struct.pack('>4L', 1, 2, 3, 4) + common.NS(struct.pack('>BL', 5, 6)))

    def test_parseRequest_window_change(self):
        """
        The payload of a window_change request is::
            uint32  columns
            uint32  rows
            uint32  x pixels
            uint32  y pixels

        parseRequest_window_change() returns (rows, columns, x pixels,
        y pixels).
        """
        self.assertEqual(session.parseRequest_window_change(struct.pack('>4L', 1, 2, 3, 4)), (2, 1, 3, 4))

    def test_packRequest_window_change(self):
        """
        See test_parseRequest_window_change for the payload format.
        """
        self.assertEqual(session.packRequest_window_change((2, 1, 3, 4)), struct.pack('>4L', 1, 2, 3, 4))