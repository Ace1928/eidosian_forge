import struct
from typing import cast, Dict, List, Tuple, Union
from maxminddb.errors import InvalidDatabaseError
from maxminddb.file import FileBuffer
from maxminddb.types import Record
def _decode_int32(self, size: int, offset: int) -> Tuple[int, int]:
    if size == 0:
        return (0, offset)
    new_offset = offset + size
    packed_bytes = self._buffer[offset:new_offset]
    if size != 4:
        packed_bytes = packed_bytes.rjust(4, b'\x00')
    value, = struct.unpack(b'!i', packed_bytes)
    return (value, new_offset)