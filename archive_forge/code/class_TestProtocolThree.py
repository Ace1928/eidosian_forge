import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
class TestProtocolThree(TestSmartProtocol):
    """Tests for v3 of the server-side protocol."""
    request_encoder = protocol.ProtocolThreeRequester
    response_decoder = protocol.ProtocolThreeDecoder
    server_protocol_class = protocol.ProtocolThreeDecoder

    def test_trivial_request(self):
        """Smoke test for the simplest possible v3 request: empty headers, no
        message parts.
        """
        output = BytesIO()
        headers = b'\x00\x00\x00\x02de'
        end = b'e'
        request_bytes = headers + end
        smart_protocol = self.server_protocol_class(LoggingMessageHandler())
        smart_protocol.accept_bytes(request_bytes)
        self.assertEqual(0, smart_protocol.next_read_size())
        self.assertEqual(b'', smart_protocol.unused_data)

    def test_repeated_excess(self):
        """Repeated calls to accept_bytes after the message end has been parsed
        accumlates the bytes in the unused_data attribute.
        """
        output = BytesIO()
        headers = b'\x00\x00\x00\x02de'
        end = b'e'
        request_bytes = headers + end
        smart_protocol = self.server_protocol_class(LoggingMessageHandler())
        smart_protocol.accept_bytes(request_bytes)
        self.assertEqual(b'', smart_protocol.unused_data)
        smart_protocol.accept_bytes(b'aaa')
        self.assertEqual(b'aaa', smart_protocol.unused_data)
        smart_protocol.accept_bytes(b'bbb')
        self.assertEqual(b'aaabbb', smart_protocol.unused_data)
        self.assertEqual(0, smart_protocol.next_read_size())

    def make_protocol_expecting_message_part(self):
        headers = b'\x00\x00\x00\x02de'
        message_handler = LoggingMessageHandler()
        smart_protocol = self.server_protocol_class(message_handler)
        smart_protocol.accept_bytes(headers)
        del message_handler.event_log[:]
        return (smart_protocol, message_handler.event_log)

    def test_decode_one_byte(self):
        """The protocol can decode a 'one byte' message part."""
        smart_protocol, event_log = self.make_protocol_expecting_message_part()
        smart_protocol.accept_bytes(b'ox')
        self.assertEqual([('byte', b'x')], event_log)

    def test_decode_bytes(self):
        """The protocol can decode a 'bytes' message part."""
        smart_protocol, event_log = self.make_protocol_expecting_message_part()
        smart_protocol.accept_bytes(b'b\x00\x00\x00\x07payload')
        self.assertEqual([('bytes', b'payload')], event_log)

    def test_decode_structure(self):
        """The protocol can decode a 'structure' message part."""
        smart_protocol, event_log = self.make_protocol_expecting_message_part()
        smart_protocol.accept_bytes(b's\x00\x00\x00\x07l3:ARGe')
        self.assertEqual([('structure', (b'ARG',))], event_log)

    def test_decode_multiple_bytes(self):
        """The protocol can decode a multiple 'bytes' message parts."""
        smart_protocol, event_log = self.make_protocol_expecting_message_part()
        smart_protocol.accept_bytes(b'b\x00\x00\x00\x05firstb\x00\x00\x00\x06second')
        self.assertEqual([('bytes', b'first'), ('bytes', b'second')], event_log)