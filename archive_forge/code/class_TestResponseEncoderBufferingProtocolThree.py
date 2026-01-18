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
class TestResponseEncoderBufferingProtocolThree(tests.TestCase):
    """Tests for buffering of responses.

    We want to avoid doing many small writes when one would do, to avoid
    unnecessary network overhead.
    """

    def setUp(self):
        super().setUp()
        self.writes = []
        self.responder = protocol.ProtocolThreeResponder(self.writes.append)

    def assertWriteCount(self, expected_count):
        self.assertEqual(expected_count, len(self.writes), 'Too many writes: %d, expected %d' % (len(self.writes), expected_count))

    def test_send_error_writes_just_once(self):
        """An error response is written to the medium all at once."""
        self.responder.send_error(Exception('An exception string.'))
        self.assertWriteCount(1)

    def test_send_response_writes_just_once(self):
        """A normal response with no body is written to the medium all at once.
        """
        response = _mod_request.SuccessfulSmartServerResponse((b'arg', b'arg'))
        self.responder.send_response(response)
        self.assertWriteCount(1)

    def test_send_response_with_body_writes_just_once(self):
        """A normal response with a monolithic body is written to the medium
        all at once.
        """
        response = _mod_request.SuccessfulSmartServerResponse((b'arg', b'arg'), body=b'body bytes')
        self.responder.send_response(response)
        self.assertWriteCount(1)

    def test_send_response_with_body_stream_buffers_writes(self):
        """A normal response with a stream body writes to the medium once."""
        response = _mod_request.SuccessfulSmartServerResponse((b'arg', b'arg'), body_stream=[b'chunk1', b'chunk2'])
        self.responder.send_response(response)
        self.assertWriteCount(3)