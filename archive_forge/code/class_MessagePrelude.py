from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
class MessagePrelude:
    """Represents the prelude of an event stream message."""

    def __init__(self, total_length, headers_length, crc):
        self.total_length = total_length
        self.headers_length = headers_length
        self.crc = crc

    @property
    def payload_length(self):
        """Calculates the total payload length.

        The extra minus 4 bytes is for the message CRC.

        :rtype: int
        :returns: The total payload length.
        """
        return self.total_length - self.headers_length - _PRELUDE_LENGTH - 4

    @property
    def payload_end(self):
        """Calculates the byte offset for the end of the message payload.

        The extra minus 4 bytes is for the message CRC.

        :rtype: int
        :returns: The byte offset from the beginning of the event stream
        message to the end of the payload.
        """
        return self.total_length - 4

    @property
    def headers_end(self):
        """Calculates the byte offset for the end of the message headers.

        :rtype: int
        :returns: The byte offset from the beginning of the event stream
        message to the end of the headers.
        """
        return _PRELUDE_LENGTH + self.headers_length