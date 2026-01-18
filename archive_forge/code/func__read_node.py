import ipaddress
import struct
from ipaddress import IPv4Address, IPv6Address
from os import PathLike
from typing import Any, AnyStr, Dict, IO, List, Optional, Tuple, Union
from maxminddb.const import MODE_AUTO, MODE_MMAP, MODE_FILE, MODE_MEMORY, MODE_FD
from maxminddb.decoder import Decoder
from maxminddb.errors import InvalidDatabaseError
from maxminddb.file import FileBuffer
from maxminddb.types import Record
def _read_node(self, node_number: int, index: int) -> int:
    base_offset = node_number * self._metadata.node_byte_size
    record_size = self._metadata.record_size
    if record_size == 24:
        offset = base_offset + index * 3
        node_bytes = b'\x00' + self._buffer[offset:offset + 3]
    elif record_size == 28:
        offset = base_offset + 3 * index
        node_bytes = bytearray(self._buffer[offset:offset + 4])
        if index:
            node_bytes[0] = 15 & node_bytes[0]
        else:
            middle = (240 & node_bytes.pop()) >> 4
            node_bytes.insert(0, middle)
    elif record_size == 32:
        offset = base_offset + index * 4
        node_bytes = self._buffer[offset:offset + 4]
    else:
        raise InvalidDatabaseError(f'Unknown record size: {record_size}')
    return struct.unpack(b'!I', node_bytes)[0]