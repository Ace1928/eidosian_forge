from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
class ReceivableBytesIO(BytesIO):
    """BytesIO with socket-like recv semantics for testing."""

    def __init__(self) -> None:
        BytesIO.__init__(self)
        self.allow_read_past_eof = False

    def recv(self, size):
        if self.tell() == len(self.getvalue()) and (not self.allow_read_past_eof):
            raise GitProtocolError('Blocking read past end of socket')
        if size == 1:
            return self.read(1)
        return self.read(size - 1)