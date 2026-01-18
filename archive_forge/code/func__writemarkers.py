import struct
import builtins
import warnings
from collections import namedtuple
def _writemarkers(self):
    if len(self._markers) == 0:
        return
    self._file.write(b'MARK')
    length = 2
    for marker in self._markers:
        id, pos, name = marker
        length = length + len(name) + 1 + 6
        if len(name) & 1 == 0:
            length = length + 1
    _write_ulong(self._file, length)
    self._marklength = length + 8
    _write_short(self._file, len(self._markers))
    for marker in self._markers:
        id, pos, name = marker
        _write_short(self._file, id)
        _write_ulong(self._file, pos)
        _write_string(self._file, name)