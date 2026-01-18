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
class SmartClientRequestProtocolTwo(SmartClientRequestProtocolOne):
    """Version two of the client side of the smart protocol.

    This prefixes the request with the value of REQUEST_VERSION_TWO.
    """
    response_marker = RESPONSE_VERSION_TWO
    request_marker = REQUEST_VERSION_TWO

    def read_response_tuple(self, expect_body=False):
        """Read a response tuple from the wire.

        This should only be called once.
        """
        version = self._request.read_line()
        if version != self.response_marker:
            self._request.finished_reading()
            raise errors.UnexpectedProtocolVersionMarker(version)
        response_status = self._request.read_line()
        result = SmartClientRequestProtocolOne._read_response_tuple(self)
        self._response_is_unknown_method(result)
        if response_status == b'success\n':
            self.response_status = True
            if not expect_body:
                self._request.finished_reading()
            return result
        elif response_status == b'failed\n':
            self.response_status = False
            self._request.finished_reading()
            raise errors.ErrorFromSmartServer(result)
        else:
            raise errors.SmartProtocolError('bad protocol status %r' % response_status)

    def _write_protocol_version(self):
        """Write any prefixes this protocol requires.

        Version two sends the value of REQUEST_VERSION_TWO.
        """
        self._request.accept_bytes(self.request_marker)

    def read_streamed_body(self):
        """Read bytes from the body, decoding into a byte stream.
        """
        _body_decoder = ChunkedBodyDecoder()
        while not _body_decoder.finished_reading:
            bytes = self._request.read_bytes(_body_decoder.next_read_size())
            if bytes == b'':
                raise errors.ConnectionReset('Connection lost while reading streamed body.')
            _body_decoder.accept_bytes(bytes)
            for body_bytes in iter(_body_decoder.read_next_chunk, None):
                if 'hpss' in debug.debug_flags and isinstance(body_bytes, str):
                    mutter('              %d byte chunk read', len(body_bytes))
                yield body_bytes
        self._request.finished_reading()