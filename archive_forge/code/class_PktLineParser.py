from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
class PktLineParser:
    """Packet line parser that hands completed packets off to a callback."""

    def __init__(self, handle_pkt) -> None:
        self.handle_pkt = handle_pkt
        self._readahead = BytesIO()

    def parse(self, data):
        """Parse a fragment of data and call back for any completed packets."""
        self._readahead.write(data)
        buf = self._readahead.getvalue()
        if len(buf) < 4:
            return
        while len(buf) >= 4:
            size = int(buf[:4], 16)
            if size == 0:
                self.handle_pkt(None)
                buf = buf[4:]
            elif size <= len(buf):
                self.handle_pkt(buf[4:size])
                buf = buf[size:]
            else:
                break
        self._readahead = BytesIO()
        self._readahead.write(buf)

    def get_tail(self):
        """Read back any unused data."""
        return self._readahead.getvalue()