from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
class DuplicateHeader(ParserError):
    """Duplicate header found in the event."""

    def __init__(self, header):
        message = 'Duplicate header present: "%s"' % header
        super().__init__(message)