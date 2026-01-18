import struct
from typing import cast, Dict, List, Tuple, Union
from maxminddb.errors import InvalidDatabaseError
from maxminddb.file import FileBuffer
from maxminddb.types import Record
def _decode_uint(self, size: int, offset: int) -> Tuple[int, int]:
    new_offset = offset + size
    uint_bytes = self._buffer[offset:new_offset]
    return (int.from_bytes(uint_bytes, 'big'), new_offset)