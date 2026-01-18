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
def _find_address_in_tree(self, packed: bytearray) -> Tuple[int, int]:
    bit_count = len(packed) * 8
    node = self._start_node(bit_count)
    node_count = self._metadata.node_count
    i = 0
    while i < bit_count and node < node_count:
        bit = 1 & packed[i >> 3] >> 7 - i % 8
        node = self._read_node(node, bit)
        i = i + 1
    if node == node_count:
        return (0, i)
    if node > node_count:
        return (node, i)
    raise InvalidDatabaseError('Invalid node in search tree')