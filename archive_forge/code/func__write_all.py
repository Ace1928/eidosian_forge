from io import BytesIO
from paramiko.common import (
from paramiko.util import ClosingContextManager, u
def _write_all(self, raw_data):
    data = memoryview(raw_data)
    while len(data) > 0:
        count = self._write(data)
        data = data[count:]
        if self._flags & self.FLAG_APPEND:
            self._size += count
            self._pos = self._realpos = self._size
        else:
            self._pos += count
            self._realpos += count
    return None