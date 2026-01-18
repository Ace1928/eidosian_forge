import _thread
import struct
import sys
import time
from collections import deque
from io import BytesIO
from fastbencode import bdecode_as_tuple, bencode
import breezy
from ... import debug, errors, osutils
from ...trace import log_exception_quietly, mutter
from . import message, request
class SmartClientRequestProtocolOne(SmartProtocolBase, Requester, message.ResponseHandler):
    """The client-side protocol for smart version 1."""

    def __init__(self, request):
        """Construct a SmartClientRequestProtocolOne.

        :param request: A SmartClientMediumRequest to serialise onto and
            deserialise from.
        """
        self._request = request
        self._body_buffer = None
        self._request_start_time = None
        self._last_verb = None
        self._headers = None

    def set_headers(self, headers):
        self._headers = dict(headers)

    def call(self, *args):
        if 'hpss' in debug.debug_flags:
            mutter('hpss call:   %s', repr(args)[1:-1])
            if getattr(self._request._medium, 'base', None) is not None:
                mutter('             (to %s)', self._request._medium.base)
            self._request_start_time = osutils.perf_counter()
        self._write_args(args)
        self._request.finished_writing()
        self._last_verb = args[0]

    def call_with_body_bytes(self, args, body):
        """Make a remote call of args with body bytes 'body'.

        After calling this, call read_response_tuple to find the result out.
        """
        if 'hpss' in debug.debug_flags:
            mutter('hpss call w/body: %s (%r...)', repr(args)[1:-1], body[:20])
            if getattr(self._request._medium, '_path', None) is not None:
                mutter('                  (to %s)', self._request._medium._path)
            mutter('              %d bytes', len(body))
            self._request_start_time = osutils.perf_counter()
            if 'hpssdetail' in debug.debug_flags:
                mutter('hpss body content: %s', body)
        self._write_args(args)
        bytes = self._encode_bulk_data(body)
        self._request.accept_bytes(bytes)
        self._request.finished_writing()
        self._last_verb = args[0]

    def call_with_body_readv_array(self, args, body):
        """Make a remote call with a readv array.

        The body is encoded with one line per readv offset pair. The numbers in
        each pair are separated by a comma, and no trailing \\n is emitted.
        """
        if 'hpss' in debug.debug_flags:
            mutter('hpss call w/readv: %s', repr(args)[1:-1])
            if getattr(self._request._medium, '_path', None) is not None:
                mutter('                  (to %s)', self._request._medium._path)
            self._request_start_time = osutils.perf_counter()
        self._write_args(args)
        readv_bytes = self._serialise_offsets(body)
        bytes = self._encode_bulk_data(readv_bytes)
        self._request.accept_bytes(bytes)
        self._request.finished_writing()
        if 'hpss' in debug.debug_flags:
            mutter('              %d bytes in readv request', len(readv_bytes))
        self._last_verb = args[0]

    def call_with_body_stream(self, args, stream):
        self._request.finished_writing()
        self._request.finished_reading()
        raise errors.UnknownSmartMethod(args[0])

    def cancel_read_body(self):
        """After expecting a body, a response code may indicate one otherwise.

        This method lets the domain client inform the protocol that no body
        will be transmitted. This is a terminal method: after calling it the
        protocol is not able to be used further.
        """
        self._request.finished_reading()

    def _read_response_tuple(self):
        result = self._recv_tuple()
        if 'hpss' in debug.debug_flags:
            if self._request_start_time is not None:
                mutter('   result:   %6.3fs  %s', osutils.perf_counter() - self._request_start_time, repr(result)[1:-1])
                self._request_start_time = None
            else:
                mutter('   result:   %s', repr(result)[1:-1])
        return result

    def read_response_tuple(self, expect_body=False):
        """Read a response tuple from the wire.

        This should only be called once.
        """
        result = self._read_response_tuple()
        self._response_is_unknown_method(result)
        self._raise_args_if_error(result)
        if not expect_body:
            self._request.finished_reading()
        return result

    def _raise_args_if_error(self, result_tuple):
        v1_error_codes = [b'norepository', b'NoSuchFile', b'FileExists', b'DirectoryNotEmpty', b'ShortReadvError', b'UnicodeEncodeError', b'UnicodeDecodeError', b'ReadOnlyError', b'nobranch', b'NoSuchRevision', b'nosuchrevision', b'LockContention', b'UnlockableTransport', b'LockFailed', b'TokenMismatch', b'ReadError', b'PermissionDenied']
        if result_tuple[0] in v1_error_codes:
            self._request.finished_reading()
            raise errors.ErrorFromSmartServer(result_tuple)

    def _response_is_unknown_method(self, result_tuple):
        """Raise UnexpectedSmartServerResponse if the response is an 'unknonwn
        method' response to the request.

        :param response: The response from a smart client call_expecting_body
            call.
        :param verb: The verb used in that call.
        :raises: UnexpectedSmartServerResponse
        """
        if result_tuple == (b'error', b"Generic bzr smart protocol error: bad request '" + self._last_verb + b"'") or result_tuple == (b'error', b"Generic bzr smart protocol error: bad request u'%s'" % self._last_verb):
            self._request.finished_reading()
            raise errors.UnknownSmartMethod(self._last_verb)

    def read_body_bytes(self, count=-1):
        """Read bytes from the body, decoding into a byte stream.

        We read all bytes at once to ensure we've checked the trailer for
        errors, and then feed the buffer back as read_body_bytes is called.
        """
        if self._body_buffer is not None:
            return self._body_buffer.read(count)
        _body_decoder = LengthPrefixedBodyDecoder()
        while not _body_decoder.finished_reading:
            bytes = self._request.read_bytes(_body_decoder.next_read_size())
            if bytes == b'':
                raise errors.ConnectionReset('Connection lost while reading response body.')
            _body_decoder.accept_bytes(bytes)
        self._request.finished_reading()
        self._body_buffer = BytesIO(_body_decoder.read_pending_data())
        if 'hpss' in debug.debug_flags:
            mutter('              %d body bytes read', len(self._body_buffer.getvalue()))
        return self._body_buffer.read(count)

    def _recv_tuple(self):
        """Receive a tuple from the medium request."""
        return _decode_tuple(self._request.read_line())

    def query_version(self):
        """Return protocol version number of the server."""
        self.call(b'hello')
        resp = self.read_response_tuple()
        if resp == (b'ok', b'1'):
            return 1
        elif resp == (b'ok', b'2'):
            return 2
        else:
            raise errors.SmartProtocolError('bad response {!r}'.format(resp))

    def _write_args(self, args):
        self._write_protocol_version()
        bytes = _encode_tuple(args)
        self._request.accept_bytes(bytes)

    def _write_protocol_version(self):
        """Write any prefixes this protocol requires.

        Version one doesn't send protocol versions.
        """