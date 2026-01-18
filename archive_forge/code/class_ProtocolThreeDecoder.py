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
class ProtocolThreeDecoder(_StatefulDecoder):
    response_marker = RESPONSE_VERSION_THREE
    request_marker = REQUEST_VERSION_THREE

    def __init__(self, message_handler, expect_version_marker=False):
        _StatefulDecoder.__init__(self)
        self._has_dispatched = False
        if expect_version_marker:
            self.state_accept = self._state_accept_expecting_protocol_version
            self._number_needed_bytes = len(MESSAGE_VERSION_THREE) + 4
        else:
            self.state_accept = self._state_accept_expecting_headers
            self._number_needed_bytes = 4
        self.decoding_failed = False
        self.request_handler = self.message_handler = message_handler

    def accept_bytes(self, bytes):
        self._number_needed_bytes = None
        try:
            _StatefulDecoder.accept_bytes(self, bytes)
        except KeyboardInterrupt:
            raise
        except SmartMessageHandlerError as exception:
            if not isinstance(exception.exc_value, errors.UnknownSmartMethod):
                log_exception_quietly()
            self.message_handler.protocol_error(exception.exc_value)
            self.accept_bytes(b'')
        except Exception as exception:
            self.decoding_failed = True
            if isinstance(exception, errors.UnexpectedProtocolVersionMarker):
                pass
            else:
                log_exception_quietly()
            self.message_handler.protocol_error(exception)

    def _extract_length_prefixed_bytes(self):
        if self._in_buffer_len < 4:
            raise _NeedMoreBytes(4)
        length, = struct.unpack('!L', self._get_in_bytes(4))
        end_of_bytes = 4 + length
        if self._in_buffer_len < end_of_bytes:
            raise _NeedMoreBytes(end_of_bytes)
        in_buf = self._get_in_buffer()
        bytes = in_buf[4:end_of_bytes]
        self._set_in_buffer(in_buf[end_of_bytes:])
        return bytes

    def _extract_prefixed_bencoded_data(self):
        prefixed_bytes = self._extract_length_prefixed_bytes()
        try:
            decoded = bdecode_as_tuple(prefixed_bytes)
        except ValueError:
            raise errors.SmartProtocolError('Bytes {!r} not bencoded'.format(prefixed_bytes))
        return decoded

    def _extract_single_byte(self):
        if self._in_buffer_len == 0:
            raise _NeedMoreBytes(1)
        in_buf = self._get_in_buffer()
        one_byte = in_buf[0:1]
        self._set_in_buffer(in_buf[1:])
        return one_byte

    def _state_accept_expecting_protocol_version(self):
        needed_bytes = len(MESSAGE_VERSION_THREE) - self._in_buffer_len
        in_buf = self._get_in_buffer()
        if needed_bytes > 0:
            if not MESSAGE_VERSION_THREE.startswith(in_buf):
                raise errors.UnexpectedProtocolVersionMarker(in_buf)
            raise _NeedMoreBytes(len(MESSAGE_VERSION_THREE))
        if not in_buf.startswith(MESSAGE_VERSION_THREE):
            raise errors.UnexpectedProtocolVersionMarker(in_buf)
        self._set_in_buffer(in_buf[len(MESSAGE_VERSION_THREE):])
        self.state_accept = self._state_accept_expecting_headers

    def _state_accept_expecting_headers(self):
        decoded = self._extract_prefixed_bencoded_data()
        if not isinstance(decoded, dict):
            raise errors.SmartProtocolError('Header object {!r} is not a dict'.format(decoded))
        self.state_accept = self._state_accept_expecting_message_part
        try:
            self.message_handler.headers_received(decoded)
        except:
            raise SmartMessageHandlerError(sys.exc_info())

    def _state_accept_expecting_message_part(self):
        message_part_kind = self._extract_single_byte()
        if message_part_kind == b'o':
            self.state_accept = self._state_accept_expecting_one_byte
        elif message_part_kind == b's':
            self.state_accept = self._state_accept_expecting_structure
        elif message_part_kind == b'b':
            self.state_accept = self._state_accept_expecting_bytes
        elif message_part_kind == b'e':
            self.done()
        else:
            raise errors.SmartProtocolError('Bad message kind byte: {!r}'.format(message_part_kind))

    def _state_accept_expecting_one_byte(self):
        byte = self._extract_single_byte()
        self.state_accept = self._state_accept_expecting_message_part
        try:
            self.message_handler.byte_part_received(byte)
        except:
            raise SmartMessageHandlerError(sys.exc_info())

    def _state_accept_expecting_bytes(self):
        prefixed_bytes = self._extract_length_prefixed_bytes()
        self.state_accept = self._state_accept_expecting_message_part
        try:
            self.message_handler.bytes_part_received(prefixed_bytes)
        except:
            raise SmartMessageHandlerError(sys.exc_info())

    def _state_accept_expecting_structure(self):
        structure = self._extract_prefixed_bencoded_data()
        self.state_accept = self._state_accept_expecting_message_part
        try:
            self.message_handler.structure_part_received(structure)
        except:
            raise SmartMessageHandlerError(sys.exc_info())

    def done(self):
        self.unused_data = self._get_in_buffer()
        self._set_in_buffer(None)
        self.state_accept = self._state_accept_reading_unused
        try:
            self.message_handler.end_received()
        except:
            raise SmartMessageHandlerError(sys.exc_info())

    def _state_accept_reading_unused(self):
        self.unused_data += self._get_in_buffer()
        self._set_in_buffer(None)

    def next_read_size(self):
        if self.state_accept == self._state_accept_reading_unused:
            return 0
        elif self.decoding_failed:
            return 0
        elif self._number_needed_bytes is not None:
            return self._number_needed_bytes - self._in_buffer_len
        else:
            raise AssertionError("don't know how many bytes are expected!")