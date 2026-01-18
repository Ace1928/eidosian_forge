import re
from io import BytesIO
from .. import errors
def _consume_line(self):
    """Take a line out of the buffer, and return the line.

        If a newline byte is not found in the buffer, the buffer is
        unchanged and this returns None instead.
        """
    newline_pos = self._buffer.find(b'\n')
    if newline_pos != -1:
        line = self._buffer[:newline_pos]
        self._buffer = self._buffer[newline_pos + 1:]
        return line
    else:
        return None