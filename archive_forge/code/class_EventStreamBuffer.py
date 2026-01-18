from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
class EventStreamBuffer:
    """Streaming based event stream buffer

    A buffer class that wraps bytes from an event stream providing parsed
    messages as they become available via an iterable interface.
    """

    def __init__(self):
        self._data = b''
        self._prelude = None
        self._header_parser = EventStreamHeaderParser()

    def add_data(self, data):
        """Add data to the buffer.

        :type data: bytes
        :param data: The bytes to add to the buffer to be used when parsing
        """
        self._data += data

    def _validate_prelude(self, prelude):
        if prelude.headers_length > _MAX_HEADERS_LENGTH:
            raise InvalidHeadersLength(prelude.headers_length)
        if prelude.payload_length > _MAX_PAYLOAD_LENGTH:
            raise InvalidPayloadLength(prelude.payload_length)

    def _parse_prelude(self):
        prelude_bytes = self._data[:_PRELUDE_LENGTH]
        raw_prelude, _ = DecodeUtils.unpack_prelude(prelude_bytes)
        prelude = MessagePrelude(*raw_prelude)
        self._validate_prelude(prelude)
        _validate_checksum(prelude_bytes[:_PRELUDE_LENGTH - 4], prelude.crc)
        return prelude

    def _parse_headers(self):
        header_bytes = self._data[_PRELUDE_LENGTH:self._prelude.headers_end]
        return self._header_parser.parse(header_bytes)

    def _parse_payload(self):
        prelude = self._prelude
        payload_bytes = self._data[prelude.headers_end:prelude.payload_end]
        return payload_bytes

    def _parse_message_crc(self):
        prelude = self._prelude
        crc_bytes = self._data[prelude.payload_end:prelude.total_length]
        message_crc, _ = DecodeUtils.unpack_uint32(crc_bytes)
        return message_crc

    def _parse_message_bytes(self):
        message_bytes = self._data[_PRELUDE_LENGTH - 4:self._prelude.payload_end]
        return message_bytes

    def _validate_message_crc(self):
        message_crc = self._parse_message_crc()
        message_bytes = self._parse_message_bytes()
        _validate_checksum(message_bytes, message_crc, crc=self._prelude.crc)
        return message_crc

    def _parse_message(self):
        crc = self._validate_message_crc()
        headers = self._parse_headers()
        payload = self._parse_payload()
        message = EventStreamMessage(self._prelude, headers, payload, crc)
        self._prepare_for_next_message()
        return message

    def _prepare_for_next_message(self):
        self._data = self._data[self._prelude.total_length:]
        self._prelude = None

    def next(self):
        """Provides the next available message parsed from the stream

        :rtype: EventStreamMessage
        :returns: The next event stream message
        """
        if len(self._data) < _PRELUDE_LENGTH:
            raise StopIteration()
        if self._prelude is None:
            self._prelude = self._parse_prelude()
        if len(self._data) < self._prelude.total_length:
            raise StopIteration()
        return self._parse_message()

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self