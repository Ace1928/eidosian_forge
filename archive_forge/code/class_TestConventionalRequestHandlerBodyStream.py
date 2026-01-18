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
class TestConventionalRequestHandlerBodyStream(tests.TestCase):
    """Tests for ConventionalRequestHandler's handling of request bodies."""

    def make_request_handler(self, request_bytes):
        """Make a ConventionalRequestHandler for the given bytes using test
        doubles for the request_handler and the responder.
        """
        from breezy.bzr.smart.message import ConventionalRequestHandler
        request_handler = InstrumentedRequestHandler()
        request_handler.response = _mod_request.SuccessfulSmartServerResponse((b'arg', b'arg'))
        responder = FakeResponder()
        message_handler = ConventionalRequestHandler(request_handler, responder)
        protocol_decoder = protocol.ProtocolThreeDecoder(message_handler)
        protocol_decoder.state_accept = protocol_decoder._state_accept_expecting_message_part
        protocol_decoder.accept_bytes(request_bytes)
        return request_handler

    def test_multiple_bytes_parts(self):
        """Each bytes part triggers a call to the request_handler's
        accept_body method.
        """
        multiple_bytes_parts = b's\x00\x00\x00\x07l3:fooeb\x00\x00\x00\x0bSome bytes\nb\x00\x00\x00\nMore bytese'
        request_handler = self.make_request_handler(multiple_bytes_parts)
        accept_body_calls = [call_info[1] for call_info in request_handler.calls if call_info[0] == 'accept_body']
        self.assertEqual([b'Some bytes\n', b'More bytes'], accept_body_calls)

    def test_error_flag_after_body(self):
        body_then_error = b's\x00\x00\x00\x07l3:fooeb\x00\x00\x00\x0bSome bytes\nb\x00\x00\x00\nMore bytesoEs\x00\x00\x00\x07l3:baree'
        request_handler = self.make_request_handler(body_then_error)
        self.assertEqual([('post_body_error_received', (b'bar',)), ('end_received',)], request_handler.calls[-2:])