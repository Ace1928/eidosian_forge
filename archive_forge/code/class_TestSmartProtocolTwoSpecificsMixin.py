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
class TestSmartProtocolTwoSpecificsMixin:

    def assertBodyStreamSerialisation(self, expected_serialisation, body_stream):
        """Assert that body_stream is serialised as expected_serialisation."""
        out_stream = BytesIO()
        protocol._send_stream(body_stream, out_stream.write)
        self.assertEqual(expected_serialisation, out_stream.getvalue())

    def assertBodyStreamRoundTrips(self, body_stream):
        """Assert that body_stream is the same after being serialised and
        deserialised.
        """
        out_stream = BytesIO()
        protocol._send_stream(body_stream, out_stream.write)
        decoder = protocol.ChunkedBodyDecoder()
        decoder.accept_bytes(out_stream.getvalue())
        decoded_stream = list(iter(decoder.read_next_chunk, None))
        self.assertEqual(body_stream, decoded_stream)

    def test_body_stream_serialisation_empty(self):
        """A body_stream with no bytes can be serialised."""
        self.assertBodyStreamSerialisation(b'chunked\nEND\n', [])
        self.assertBodyStreamRoundTrips([])

    def test_body_stream_serialisation(self):
        stream = [b'chunk one', b'chunk two', b'chunk three']
        self.assertBodyStreamSerialisation(b'chunked\n' + b'9\nchunk one' + b'9\nchunk two' + b'b\nchunk three' + b'END\n', stream)
        self.assertBodyStreamRoundTrips(stream)

    def test_body_stream_with_empty_element_serialisation(self):
        """A body stream can include ''.

        The empty string can be transmitted like any other string.
        """
        stream = [b'', b'chunk']
        self.assertBodyStreamSerialisation(b'chunked\n' + b'0\n' + b'5\nchunk' + b'END\n', stream)
        self.assertBodyStreamRoundTrips(stream)

    def test_body_stream_error_serialistion(self):
        stream = [b'first chunk', _mod_request.FailedSmartServerResponse((b'FailureName', b'failure arg'))]
        expected_bytes = b'chunked\n' + b'b\nfirst chunk' + b'ERR\n' + b'b\nFailureName' + b'b\nfailure arg' + b'END\n'
        self.assertBodyStreamSerialisation(expected_bytes, stream)
        self.assertBodyStreamRoundTrips(stream)

    def test__send_response_includes_failure_marker(self):
        """FailedSmartServerResponse have 'failed
' after the version."""
        out_stream = BytesIO()
        smart_protocol = protocol.SmartServerRequestProtocolTwo(None, out_stream.write)
        smart_protocol._send_response(_mod_request.FailedSmartServerResponse((b'x',)))
        self.assertEqual(protocol.RESPONSE_VERSION_TWO + b'failed\nx\n', out_stream.getvalue())

    def test__send_response_includes_success_marker(self):
        """SuccessfulSmartServerResponse have 'success
' after the version."""
        out_stream = BytesIO()
        smart_protocol = protocol.SmartServerRequestProtocolTwo(None, out_stream.write)
        smart_protocol._send_response(_mod_request.SuccessfulSmartServerResponse((b'x',)))
        self.assertEqual(protocol.RESPONSE_VERSION_TWO + b'success\nx\n', out_stream.getvalue())

    def test__send_response_with_body_stream_sets_finished_reading(self):
        smart_protocol = protocol.SmartServerRequestProtocolTwo(None, lambda x: None)
        self.assertEqual(1, smart_protocol.next_read_size())
        smart_protocol._send_response(_mod_request.SuccessfulSmartServerResponse((b'x',), body_stream=[]))
        self.assertEqual(0, smart_protocol.next_read_size())

    def test_streamed_body_bytes(self):
        body_header = b'chunked\n'
        two_body_chunks = b'4\n1234' + b'3\n567'
        body_terminator = b'END\n'
        server_bytes = protocol.RESPONSE_VERSION_TWO + b'success\nok\n' + body_header + two_body_chunks + body_terminator
        input = BytesIO(server_bytes)
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        request = client_medium.get_request()
        smart_protocol = protocol.SmartClientRequestProtocolTwo(request)
        smart_protocol.call(b'foo')
        smart_protocol.read_response_tuple(True)
        stream = smart_protocol.read_streamed_body()
        self.assertEqual([b'1234', b'567'], list(stream))

    def test_read_streamed_body_error(self):
        """When a stream is interrupted by an error..."""
        body_header = b'chunked\n'
        a_body_chunk = b'4\naaaa'
        err_signal = b'ERR\n'
        err_chunks = b'a\nerror arg1' + b'4\narg2'
        finish = b'END\n'
        body = body_header + a_body_chunk + err_signal + err_chunks + finish
        server_bytes = protocol.RESPONSE_VERSION_TWO + b'success\nok\n' + body
        input = BytesIO(server_bytes)
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        smart_request = client_medium.get_request()
        smart_protocol = protocol.SmartClientRequestProtocolTwo(smart_request)
        smart_protocol.call(b'foo')
        smart_protocol.read_response_tuple(True)
        expected_chunks = [b'aaaa', _mod_request.FailedSmartServerResponse((b'error arg1', b'arg2'))]
        stream = smart_protocol.read_streamed_body()
        self.assertEqual(expected_chunks, list(stream))

    def test_streamed_body_bytes_interrupted_connection(self):
        body_header = b'chunked\n'
        incomplete_body_chunk = b'9999\nincomplete chunk'
        server_bytes = protocol.RESPONSE_VERSION_TWO + b'success\nok\n' + body_header + incomplete_body_chunk
        input = BytesIO(server_bytes)
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        request = client_medium.get_request()
        smart_protocol = protocol.SmartClientRequestProtocolTwo(request)
        smart_protocol.call(b'foo')
        smart_protocol.read_response_tuple(True)
        stream = smart_protocol.read_streamed_body()
        self.assertRaises(errors.ConnectionReset, next, stream)

    def test_client_read_response_tuple_sets_response_status(self):
        server_bytes = protocol.RESPONSE_VERSION_TWO + b'success\nok\n'
        input = BytesIO(server_bytes)
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        request = client_medium.get_request()
        smart_protocol = protocol.SmartClientRequestProtocolTwo(request)
        smart_protocol.call(b'foo')
        smart_protocol.read_response_tuple(False)
        self.assertEqual(True, smart_protocol.response_status)

    def test_client_read_response_tuple_raises_UnknownSmartMethod(self):
        """read_response_tuple raises UnknownSmartMethod if the response says
        the server did not recognise the request.
        """
        server_bytes = protocol.RESPONSE_VERSION_TWO + b'failed\n' + b"error\x01Generic bzr smart protocol error: bad request 'foo'\n"
        input = BytesIO(server_bytes)
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        request = client_medium.get_request()
        smart_protocol = protocol.SmartClientRequestProtocolTwo(request)
        smart_protocol.call(b'foo')
        self.assertRaises(errors.UnknownSmartMethod, smart_protocol.read_response_tuple)
        self.assertRaises(errors.ReadingCompleted, smart_protocol.read_body_bytes)