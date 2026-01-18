from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
class NoInitialResponseError(ParserError):
    """An event of type initial-response was not received.

    This exception is raised when the event stream produced no events or
    the first event in the stream was not of the initial-response type.
    """

    def __init__(self):
        message = 'First event was not of the initial-response type'
        super().__init__(message)