from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
class MessageHandler:
    """Base class for handling messages received via the smart protocol.

    As parts of a message are received, the corresponding PART_received method
    will be called.
    """

    def __init__(self):
        self.headers = None

    def headers_received(self, headers):
        """Called when message headers are received.

        This default implementation just stores them in self.headers.
        """
        self.headers = headers

    def byte_part_received(self, byte):
        """Called when a 'byte' part is received.

        Note that a 'byte' part is a message part consisting of exactly one
        byte.
        """
        raise NotImplementedError(self.byte_received)

    def bytes_part_received(self, bytes):
        """Called when a 'bytes' part is received.

        A 'bytes' message part can contain any number of bytes.  It should not
        be confused with a 'byte' part, which is always a single byte.
        """
        raise NotImplementedError(self.bytes_received)

    def structure_part_received(self, structure):
        """Called when a 'structure' part is received.

        :param structure: some structured data, which will be some combination
            of list, dict, int, and str objects.
        """
        raise NotImplementedError(self.bytes_received)

    def protocol_error(self, exception):
        """Called when there is a protocol decoding error.

        The default implementation just re-raises the exception.
        """
        raise

    def end_received(self):
        """Called when the end of the message is received."""
        pass