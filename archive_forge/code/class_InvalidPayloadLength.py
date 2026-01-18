from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
class InvalidPayloadLength(ParserError):
    """Payload length is longer than the maximum."""

    def __init__(self, length):
        message = 'Payload length of {} exceeded the maximum of {}'.format(length, _MAX_PAYLOAD_LENGTH)
        super().__init__(message)