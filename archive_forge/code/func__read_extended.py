import struct
from typing import cast, Dict, List, Tuple, Union
from maxminddb.errors import InvalidDatabaseError
from maxminddb.file import FileBuffer
from maxminddb.types import Record
def _read_extended(self, offset: int) -> Tuple[int, int]:
    next_byte = self._buffer[offset]
    type_num = next_byte + 7
    if type_num < 7:
        raise InvalidDatabaseError(f'Something went horribly wrong in the decoder. An extended type resolved to a type number < 8 ({type_num})')
    return (type_num, offset + 1)