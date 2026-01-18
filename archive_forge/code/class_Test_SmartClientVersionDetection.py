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
class Test_SmartClientVersionDetection(tests.TestCase):
    """Tests for _SmartClient's automatic protocol version detection.

    On the first remote call, _SmartClient will keep retrying the request with
    different protocol versions until it finds one that works.
    """

    def test_version_three_server(self):
        """With a protocol 3 server, only one request is needed."""
        medium = MockMedium()
        smart_client = client._SmartClient(medium, headers={})
        message_start = protocol.MESSAGE_VERSION_THREE + b'\x00\x00\x00\x02de'
        medium.expect_request(message_start + b's\x00\x00\x00\x1el11:method-name5:arg 15:arg 2ee', message_start + b's\x00\x00\x00\x13l14:response valueee')
        result = smart_client.call(b'method-name', b'arg 1', b'arg 2')
        self.assertEqual((b'response value',), result)
        self.assertEqual([], medium._expected_events)
        self.assertFalse(medium._is_remote_before((1, 6)))

    def test_version_two_server(self):
        """If the server only speaks protocol 2, the client will first try
        version 3, then fallback to protocol 2.

        Further, _SmartClient caches the detection, so future requests will all
        use protocol 2 immediately.
        """
        medium = MockMedium()
        smart_client = client._SmartClient(medium, headers={})
        medium.expect_request(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02de' + b's\x00\x00\x00\x1el11:method-name5:arg 15:arg 2ee', b'bzr response 2\nfailed\n\n')
        medium.expect_disconnect()
        medium.expect_request(b'bzr request 2\nmethod-name\x01arg 1\x01arg 2\n', b'bzr response 2\nsuccess\nresponse value\n')
        result = smart_client.call(b'method-name', b'arg 1', b'arg 2')
        self.assertEqual((b'response value',), result)
        medium.expect_request(b'bzr request 2\nanother-method\n', b'bzr response 2\nsuccess\nanother response\n')
        result = smart_client.call(b'another-method')
        self.assertEqual((b'another response',), result)
        self.assertEqual([], medium._expected_events)
        self.assertTrue(medium._is_remote_before((1, 6)))

    def test_unknown_version(self):
        """If the server does not use any known (or at least supported)
        protocol version, a SmartProtocolError is raised.
        """
        medium = MockMedium()
        smart_client = client._SmartClient(medium, headers={})
        unknown_protocol_bytes = b'Unknown protocol!'
        medium.expect_request(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02de' + b's\x00\x00\x00\x1el11:method-name5:arg 15:arg 2ee', unknown_protocol_bytes)
        medium.expect_disconnect()
        medium.expect_request(b'bzr request 2\nmethod-name\x01arg 1\x01arg 2\n', unknown_protocol_bytes)
        medium.expect_disconnect()
        self.assertRaises(errors.SmartProtocolError, smart_client.call, b'method-name', b'arg 1', b'arg 2')
        self.assertEqual([], medium._expected_events)

    def test_first_response_is_error(self):
        """If the server replies with an error, then the version detection
        should be complete.

        This test is very similar to test_version_two_server, but catches a bug
        we had in the case where the first reply was an error response.
        """
        medium = MockMedium()
        smart_client = client._SmartClient(medium, headers={})
        message_start = protocol.MESSAGE_VERSION_THREE + b'\x00\x00\x00\x02de'
        medium.expect_request(message_start + b's\x00\x00\x00\x10l11:method-nameee', b'bzr response 2\nfailed\n\n')
        medium.expect_disconnect()
        medium.expect_request(b'bzr request 2\nmethod-name\n', b'bzr response 2\nfailed\nFooBarError\n')
        err = self.assertRaises(errors.ErrorFromSmartServer, smart_client.call, b'method-name')
        self.assertEqual((b'FooBarError',), err.error_tuple)
        medium.expect_request(b'bzr request 2\nmethod-name\n', b'bzr response 2\nsuccess\nresponse value\n')
        result = smart_client.call(b'method-name')
        self.assertEqual((b'response value',), result)
        self.assertEqual([], medium._expected_events)