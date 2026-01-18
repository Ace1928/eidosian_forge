import array
import contextlib
import enum
import struct
def _WriteOffset(self, offset, byte_width):
    relative_offset = len(self._buf) - offset
    assert byte_width == 8 or relative_offset < 1 << 8 * byte_width
    self._Write(U, relative_offset, byte_width)